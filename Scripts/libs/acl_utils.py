# -*- coding: utf-8 -*-
"""ACL（Ascend Computing Language）Python 推理封装层。

提供：
- init_acl / release_acl：全局 ACL 初始化与释放（进程内只初始化一次）
- OmModel：单个 .om 离线模型的加载、推理、资源释放

设计说明：
- 输入：list[np.ndarray]（每个数组按模型输入节点顺序对应，需为连续内存）
- 输出：list[np.ndarray]（按输出节点 dtype 还原为一维数组，由上层 reshape）
  上层（det_infer / pose_infer）已知输出形状，自行 reshape 更直观可靠。
"""

import os
import time
import logging
import numpy as np

try:
    import acl
except ImportError as e:
    raise ImportError(
        "未找到 acl 模块，请先 source 昇腾环境：\n"
        "  source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        f"原始错误：{e}"
    )

logger = logging.getLogger("acl_utils")

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_NORMAL_ONLY = 0

_ACL_DTYPE_TO_NP = {
    0: np.float32,   # ACL_FLOAT32
    1: np.float16,   # ACL_FLOAT16
    2: np.int8,      # ACL_INT8
    3: np.int32,     # ACL_INT32
    4: np.uint8,     # ACL_UINT8
}


class _AclRuntime:
    """进程级单例，保证 acl.init / set_device / context 只创建一次。"""

    _initialized = False
    _context = None
    _device_id = 0
    _ref_count = 0

    @classmethod
    def init(cls, device_id=0):
        if cls._initialized:
            cls._ref_count += 1
            return
        ret = acl.init()
        assert ret in (0, 507008), f"acl.init 失败, ret={ret}"
        cls._device_id = device_id
        ret = acl.rt.set_device(device_id)
        assert ret == 0, f"acl.rt.set_device 失败, ret={ret}"
        cls._context, ret = acl.rt.create_context(device_id)
        assert ret == 0, f"acl.rt.create_context 失败, ret={ret}"
        cls._initialized = True
        cls._ref_count = 1
        logger.info("ACL 初始化完成, device_id=%d", device_id)

    @classmethod
    def release(cls):
        if not cls._initialized:
            return
        cls._ref_count -= 1
        if cls._ref_count > 0:
            return
        if cls._context is not None:
            acl.rt.destroy_context(cls._context)
            cls._context = None
        acl.rt.reset_device(cls._device_id)
        acl.finalize()
        cls._initialized = False
        logger.info("ACL 资源已释放")

    @classmethod
    def set_current_context(cls):
        """把当前线程绑定到本进程的 ACL context。

        ACL 要求每个线程在调用 acl.mdl.execute 前先设置当前 context
        （主线程在 create_context 后已隐式绑定，其余线程必须显式调用）。
        异步多线程流水线（读图/检测/关键点）的 worker 线程启动时须调用。
        """
        if not cls._initialized or cls._context is None:
            return
        ret = acl.rt.set_context(cls._context)
        if ret != 0:
            raise RuntimeError(f"acl.rt.set_context 失败, ret={ret}")


def init_acl(device_id=0):
    _AclRuntime.init(device_id)


def release_acl():
    _AclRuntime.release()


def bind_thread_context():
    """把当前线程绑定到 ACL context，供异步 worker 线程调用。"""
    _AclRuntime.set_current_context()


def _np_dtype_of_acl(acl_dtype):
    return _ACL_DTYPE_TO_NP.get(acl_dtype, np.float32)


class OmModel:
    """单个 .om 模型的封装。

    典型用法：
        init_acl(0)
        model = OmModel("Model/RTMDet_Tiny.om")
        outputs = model.infer([input_array])   # outputs: list[np.ndarray]
        ...
        model.close()
        release_acl()
    """

    def __init__(self, model_path, device_id=0):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"om 模型不存在: {model_path}")
        init_acl(device_id)
        self.device_id = device_id
        self.model_path = model_path
        self._closed = False

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        assert ret == 0, f"加载模型失败: {model_path}, ret={ret}"

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        assert ret == 0, f"get_desc 失败, ret={ret}"

        self.input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num = acl.mdl.get_num_outputs(self.model_desc)

        self._input_info = self._collect_io_info(is_input=True)
        self._output_info = self._collect_io_info(is_input=False)

        self.input_dataset = self._create_dataset(self._input_info)
        self.output_dataset = self._create_dataset(self._output_info)

        logger.info("已加载模型: %s (inputs=%d, outputs=%d)",
                    os.path.basename(model_path), self.input_num, self.output_num)
        for i, info in enumerate(self._input_info):
            logger.debug("  input[%d] size=%d dtype=%s", i, info["size"], info["np_dtype"])
        for i, info in enumerate(self._output_info):
            logger.debug("  output[%d] size=%d dtype=%s", i, info["size"], info["np_dtype"])

    def _collect_io_info(self, is_input):
        infos = []
        n = self.input_num if is_input else self.output_num
        for i in range(n):
            if is_input:
                size = acl.mdl.get_input_size_by_index(self.model_desc, i)
                dtype = acl.mdl.get_input_data_type(self.model_desc, i)
            else:
                size = acl.mdl.get_output_size_by_index(self.model_desc, i)
                dtype = acl.mdl.get_output_data_type(self.model_desc, i)
            infos.append({
                "size": int(size),
                "acl_dtype": int(dtype),
                "np_dtype": _np_dtype_of_acl(int(dtype)),
            })
        return infos

    def _create_dataset(self, infos):
        dataset = acl.mdl.create_dataset()
        for info in infos:
            buf, ret = acl.rt.malloc(info["size"], ACL_MEM_MALLOC_NORMAL_ONLY)
            assert ret == 0, f"acl.rt.malloc 失败, ret={ret}"
            data_buf = acl.create_data_buffer(buf, info["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buf)
            assert ret == 0, f"add_dataset_buffer 失败, ret={ret}"
            info["device_ptr"] = buf
            info["data_buf"] = data_buf
        return dataset

    @staticmethod
    def _host_to_device(host_arr, device_ptr, size):
        """host_arr: 连续 np.ndarray -> device 内存。"""
        host_ptr = acl.util.numpy_to_ptr(host_arr)
        ret = acl.rt.memcpy(device_ptr, size, host_ptr, size,
                            ACL_MEMCPY_HOST_TO_DEVICE)
        assert ret == 0, f"H2D memcpy 失败, ret={ret}"

    @staticmethod
    def _device_to_host(device_ptr, size, np_dtype):
        """device 内存 -> host np.ndarray（一维，按 np_dtype 还原）。"""
        out = np.zeros(size, dtype=np.uint8)
        host_ptr = acl.util.numpy_to_ptr(out)
        ret = acl.rt.memcpy(host_ptr, size, device_ptr, size,
                            ACL_MEMCPY_DEVICE_TO_HOST)
        assert ret == 0, f"D2H memcpy 失败, ret={ret}"
        return np.frombuffer(out.tobytes(), dtype=np_dtype)

    def infer(self, inputs):
        """执行推理。

        Args:
            inputs: list[np.ndarray]，长度等于输入节点数，每个数组形状与模型输入一致。
        Returns:
            list[np.ndarray]：每个输出节点对应一个一维 np.ndarray（按其 dtype 还原），
            上层按已知形状 reshape。

        计时信息：推理后读取实例属性
            self.last_total_ms   整体 infer 耗时
            self.last_h2d_ms     H2D 拷贝耗时
            self.last_exec_ms    acl.mdl.execute 耗时（算子/模型执行）
            self.last_d2h_ms     D2H 拷贝耗时
        """
        if self._closed:
            raise RuntimeError("模型已关闭")
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if len(inputs) != self.input_num:
            raise ValueError(
                f"输入数量不匹配: 期望 {self.input_num}, 实际 {len(inputs)}")

        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        for i, inp in enumerate(inputs):
            arr = np.ascontiguousarray(inp, dtype=self._input_info[i]["np_dtype"])
            expect_size = self._input_info[i]["size"]
            if arr.nbytes != expect_size:
                raise ValueError(
                    f"input[{i}] 字节数不匹配: 期望 {expect_size}, 实际 {arr.nbytes} "
                    f"(shape={arr.shape}, dtype={arr.dtype})")
            self._host_to_device(arr,
                                 self._input_info[i]["device_ptr"], expect_size)
        self.last_h2d_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        assert ret == 0, f"acl.mdl.execute 失败, ret={ret}"
        self.last_exec_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        outputs = []
        for i, info in enumerate(self._output_info):
            out_flat = self._device_to_host(info["device_ptr"], info["size"],
                                            info["np_dtype"])
            outputs.append(out_flat)
        self.last_d2h_ms = (time.perf_counter() - t0) * 1000

        self.last_total_ms = (time.perf_counter() - t_total0) * 1000
        return outputs

    def output_shapes(self):
        """尝试获取每个输出节点的形状（部分 ACL 版本支持）。失败返回 None 列表。"""
        shapes = []
        for i in range(self.output_num):
            try:
                dims, ret = acl.mdl.get_output_dims(self.model_desc, i)
                if ret == 0:
                    shapes.append(list(dims))
                    continue
            except Exception:
                pass
            shapes.append(None)
        return shapes

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            acl.mdl.destroy_dataset(self.input_dataset)
            acl.mdl.destroy_dataset(self.output_dataset)
            for info in self._input_info + self._output_info:
                if info.get("data_buf") is not None:
                    acl.destroy_data_buffer(info["data_buf"])
                    info["data_buf"] = None
                if info.get("device_ptr") is not None:
                    acl.rt.free(info["device_ptr"])
                    info["device_ptr"] = None
            acl.mdl.destroy_desc(self.model_desc)
            acl.mdl.unload(self.model_id)
            logger.info("已卸载模型: %s", os.path.basename(self.model_path))
        except Exception as e:
            logger.warning("释放模型资源时异常: %s", e)
        release_acl()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
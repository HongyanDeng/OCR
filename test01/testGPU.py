import paddle


print(paddle.__version__)          # 应输出 3.1.0
print(paddle.device.is_compiled_with_cuda())  # 应输出 True
print(paddle.device.get_device())  # 打印当前设备（如 gpu:0）
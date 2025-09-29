import asyncio


async def run_command(command: str):
    """
    异步运行命令行命令
    """
    print(f"Running command: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True  # 适用于 Windows CMD 命令
    )
    stdout, stderr = await process.communicate()

    # 捕获输出并解码，指定正确的编码
    if stdout:
        try:
            print(f"[stdout]\n{stdout.decode('GBK')}")
        except UnicodeDecodeError:
            print(f"[stdout]\n{stdout.decode('utf-8', errors='replace')}")

    if stderr:
        try:
            print(f"[stderr]\n{stderr.decode('GBK')}")
        except UnicodeDecodeError:
            print(f"[stderr]\n{stderr.decode('utf-8', errors='replace')}")
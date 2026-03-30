import os

# 精准指向你的 libhsl.dll
dll_path = r"D:\microsoft_downloads\Ipopt-3.14.19-win64-msvs2022-md\Ipopt-3.14.19-win64-msvs2022-md\bin\libhsl.dll"

print(f"开始对 {dll_path} 进行二进制手术...")

# 1. 读取原始二进制数据
with open(dll_path, "rb") as f:
    data = bytearray(f.read())

# 2. 备份原文件 (防呆机制)
with open(dll_path + ".bak", "wb") as f:
    f.write(data)
print("已生成备份文件：libhsl.dll.bak")

# 3. IPOPT 需要的 MA57 和 MA27 核心函数列表
symbols = [
    b"ma57ad", b"ma57bd", b"ma57cd", b"ma57ed", b"ma57id",
    b"ma27ad", b"ma27bd", b"ma27cd"
]

# 4. 实施精准替换
count = 0
for sym in symbols:
    old_sym = sym + b"_"     # MinGW 导出的名字 (带下划线)
    new_sym = sym + b"\x00"  # MSVC 期待的名字 (\x00 是 C 语言的字符串终止符)
    
    if old_sym in data:
        data = data.replace(old_sym, new_sym)
        count += 1
        print(f"✅ 成功修复符号: {old_sym.decode()} -> {sym.decode()}")

# 5. 覆盖写回
with open(dll_path, "wb") as f:
    f.write(data)

print(f"\n🎉 手术完成！共修改了 {count} 处二进制特征。")
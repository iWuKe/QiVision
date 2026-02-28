"""
IDA Python 脚本 — 导出 find_shape_model 缺失的函数和代码段
==========================================================

使用方法:
  1. 在 IDA Pro 中打开目标 DLL
  2. File -> Script file... -> 选择此脚本
  3. 输出文件保存到当前目录: exported_find_missing.txt

导出内容:
  Part 1: 3 个优先级最高的缺失子函数
    - sub_1800894C0
    - sub_18005BE10
    - sub_18007C9F0

  Part 2: 2 个纯汇编 SIMD 核心 (反汇编输出)
    - sub_1800360C0 (getNormal33 SSE, 3326B)
    - sub_180037130 (getNormal34 AVX, 3287B)

  Part 3: 3 个未确认常量值
    - 0x1800D6BD0
    - 0x1800D6BF8
    - 0x1800D6B48

  Part 4: find_shape_model 主函数中 4 个"省略"段的完整代码
    - 超时检查 (line 1334 附近)
    - 超时检查 (line 1383 附近)
    - subPixelMode==2 完整循环体 (line 1497 附近)
    - 计时/校验 + 结果输出完整代码 (line 1630 附近)
"""

import idaapi
import idautils
import idc
import ida_hexrays
import ida_bytes
import struct

import os
OUTPUT_FILE = os.path.join(os.path.expanduser("~"), "Desktop", "exported_find_missing.txt")

# ============================================================
# 配置
# ============================================================

# Part 1: 需要反编译的子函数
DECOMPILE_FUNCS = [
    (0x1800894C0, "sub_1800894C0", "优先级 #1"),
    (0x18005BE10, "sub_18005BE10", "优先级 #2"),
    (0x18007C9F0, "sub_18007C9F0", "优先级 #3"),
]

# Part 2: 纯汇编函数 (Hex-Rays 无法反编译，导出反汇编)
ASM_FUNCS = [
    (0x1800360C0, "sub_1800360C0", "getNormal33 SSE 核心", 3326),
    (0x180037130, "sub_180037130", "getNormal34 AVX 核心", 3287),
]

# Part 3: 未确认常量
UNKNOWN_CONSTANTS = [
    (0x1800D6BD0, "float",  "梯度范数缩放 / 归一化, 可能是 PI 或 4.0f"),
    (0x1800D6BF8, "float",  "sigma^2 系数 (搜索窗口)"),
    (0x1800D6B48, "double", "坐标偏移系数, 可能是 0.5"),
]

# Part 4: find_shape_model 主函数中需要完整导出的地址范围
# 我们直接反编译整个 find_shape_model，然后从输出中提取
FIND_SHAPE_MODEL_ADDR = 0x18008A1A0  # find_shape_model 入口地址
# 如果上面的地址不对，脚本会尝试搜索

# ============================================================
# 工具函数
# ============================================================

def get_func_decompiled(ea):
    """反编译指定地址的函数，返回伪代码字符串"""
    try:
        cfunc = ida_hexrays.decompile(ea)
        if cfunc:
            return str(cfunc)
    except ida_hexrays.DecompilationFailure as e:
        return f"[反编译失败: {e}]"
    except Exception as e:
        return f"[反编译异常: {e}]"
    return "[反编译返回 None]"


def get_func_disasm(ea, max_size=0x2000):
    """获取函数的反汇编输出"""
    func = idaapi.get_func(ea)
    if not func:
        return f"[无法获取函数 @ 0x{ea:X}]"

    lines = []
    current = func.start_ea
    end = min(func.end_ea, func.start_ea + max_size)

    while current < end:
        disasm = idc.generate_disasm_line(current, 0)
        if disasm:
            lines.append(f"  {current:016X}  {disasm}")
        size = idc.get_item_size(current)
        if size == 0:
            break
        current += size

    return "\n".join(lines)


def read_bytes_at(ea, size):
    """读取指定地址的原始字节"""
    return ida_bytes.get_bytes(ea, size)


def read_float(ea):
    """读取 float (4 bytes)"""
    data = read_bytes_at(ea, 4)
    if data and len(data) == 4:
        val = struct.unpack('<f', data)[0]
        return val, data.hex()
    return None, None


def read_double(ea):
    """读取 double (8 bytes)"""
    data = read_bytes_at(ea, 8)
    if data and len(data) == 8:
        val = struct.unpack('<d', data)[0]
        return val, data.hex()
    return None, None


def read_xmmword(ea):
    """读取 128-bit xmmword (16 bytes)"""
    data = read_bytes_at(ea, 16)
    if data and len(data) == 16:
        return data.hex()
    return None


def read_ymmword(ea):
    """读取 256-bit ymmword (32 bytes)"""
    data = read_bytes_at(ea, 32)
    if data and len(data) == 32:
        return data.hex()
    return None


def find_func_by_name(name):
    """按名称搜索函数地址"""
    for ea in idautils.Functions():
        if idc.get_func_name(ea) == name:
            return ea
    return None


def get_func_size(ea):
    """获取函数大小"""
    func = idaapi.get_func(ea)
    if func:
        return func.end_ea - func.start_ea
    return 0

# ============================================================
# 主导出逻辑
# ============================================================

def main():
    out = []

    def w(s=""):
        out.append(s)

    w("=" * 80)
    w("find_shape_model 缺失内容导出")
    w("=" * 80)
    w()

    # ----------------------------------------------------------
    # Part 1: 反编译缺失子函数
    # ----------------------------------------------------------
    w("=" * 80)
    w("Part 1: 缺失子函数反编译 (Hex-Rays 伪代码)")
    w("=" * 80)
    w()

    for addr, name, priority in DECOMPILE_FUNCS:
        w(f"--- {name} ({priority}) ---")
        w(f"地址: 0x{addr:X}")
        func_size = get_func_size(addr)
        w(f"函数大小: {func_size} bytes")
        w()
        w("```c")
        pseudo = get_func_decompiled(addr)
        w(pseudo)
        w("```")
        w()

        # 同时导出该函数调用的所有子函数列表
        w(f"--- {name} 调用的子函数 ---")
        func = idaapi.get_func(addr)
        if func:
            called = set()
            for head in idautils.FuncItems(addr):
                for xref in idautils.CodeRefsFrom(head, 0):
                    target_func = idaapi.get_func(xref)
                    if target_func and target_func.start_ea != addr:
                        called.add(target_func.start_ea)
            for callee in sorted(called):
                callee_name = idc.get_func_name(callee)
                callee_size = get_func_size(callee)
                w(f"  调用: {callee_name} (0x{callee:X}, {callee_size}B)")
        w()
        w()

    # ----------------------------------------------------------
    # Part 2: 纯汇编 SIMD 核心函数
    # ----------------------------------------------------------
    w("=" * 80)
    w("Part 2: 纯汇编 SIMD 核心 (反汇编)")
    w("=" * 80)
    w()

    for addr, name, desc, expected_size in ASM_FUNCS:
        w(f"--- {name} ({desc}, 预期 {expected_size}B) ---")
        w(f"地址: 0x{addr:X}")
        actual_size = get_func_size(addr)
        w(f"实际大小: {actual_size} bytes")
        w()

        # 先尝试反编译，如果失败则输出反汇编
        w("尝试反编译:")
        w("```c")
        pseudo = get_func_decompiled(addr)
        w(pseudo)
        w("```")
        w()

        w("反汇编输出:")
        w("```asm")
        disasm = get_func_disasm(addr, max_size=expected_size + 256)
        w(disasm)
        w("```")
        w()
        w()

    # ----------------------------------------------------------
    # Part 3: 未确认常量值
    # ----------------------------------------------------------
    w("=" * 80)
    w("Part 3: 未确认常量值")
    w("=" * 80)
    w()

    for addr, type_str, desc in UNKNOWN_CONSTANTS:
        w(f"--- 0x{addr:X} ({type_str}) ---")
        w(f"语义: {desc}")

        if type_str == "float":
            val, hex_str = read_float(addr)
            if val is not None:
                w(f"值: {val} (hex: {hex_str})")
            else:
                w(f"[读取失败]")
            # 额外读取周围上下文 (前后各 16 字节)
            w(f"上下文 (±16B):")
            ctx = read_bytes_at(addr - 16, 48)
            if ctx:
                w(f"  [{addr-16:X}]: {ctx[:16].hex()}")
                w(f"  [{addr:X}]:     {ctx[16:20].hex()} <-- 目标")
                w(f"  [{addr+4:X}]:   {ctx[20:48].hex()}")

        elif type_str == "double":
            val, hex_str = read_double(addr)
            if val is not None:
                w(f"值: {val} (hex: {hex_str})")
            else:
                w(f"[读取失败]")
            ctx = read_bytes_at(addr - 16, 48)
            if ctx:
                w(f"上下文 (±16B):")
                w(f"  [{addr-16:X}]: {ctx[:16].hex()}")
                w(f"  [{addr:X}]:     {ctx[16:24].hex()} <-- 目标")
                w(f"  [{addr+8:X}]:   {ctx[24:48].hex()}")

        w()

    # 额外: 导出所有已知 SIMD 常量的实际字节
    w("--- 已知 SIMD 常量实际字节 ---")
    simd_constants = [
        (0x1800D6E80, "xmmword", "double 符号翻转 mask"),
        (0x1800D6D90, "xmmword", "float 取整调整量"),
        (0x1800D6D10, "xmmword", "梯度符号提取 (vpxor)"),
        (0x1800D6D30, "xmmword", "int16 sign -> int8 pack"),
        (0x1800D7420, "ymmword", "float abs mask (AVX2)"),
        (0x1800D7460, "ymmword", "float sign mask (AVX2)"),
        (0x1800D6F40, "ymmword", "sign->{+1,-1} 常量"),
        (0x1800D6EC0, "ymmword", "shuffle/pack mask"),
        (0x1800D6F60, "ymmword", "shuffle mask 备用"),
        (0x1800D6F80, "ymmword", "int32 sign -> byte 重排"),
    ]
    for addr, type_str, desc in simd_constants:
        if type_str == "xmmword":
            data = read_xmmword(addr)
            size_str = "16B"
        else:
            data = read_ymmword(addr)
            size_str = "32B"
        w(f"  0x{addr:X} ({size_str}, {desc}): {data if data else '[读取失败]'}")
    w()
    w()

    # ----------------------------------------------------------
    # Part 4: find_shape_model 完整反编译 (用于补充省略段)
    # ----------------------------------------------------------
    w("=" * 80)
    w("Part 4: find_shape_model 主函数完整反编译")
    w("  (用于替换文档中 4 处 '省略' 标记)")
    w("  - line 1334: 超时检查")
    w("  - line 1383: 超时检查")
    w("  - line 1497: subPixelMode==2 完整循环体")
    w("  - line 1630: 计时/校验 + 结果输出")
    w("=" * 80)
    w()

    # 尝试找到 find_shape_model
    find_addr = FIND_SHAPE_MODEL_ADDR
    func = idaapi.get_func(find_addr)
    if not func:
        # 尝试搜索
        find_addr = find_func_by_name("find_shape_model")
        if find_addr is None:
            # 尝试其他可能的名称
            for name_try in ["_find_shape_model", "find_shape_model",
                             "sub_18008A1A0", "sub_1800894C0"]:
                find_addr = find_func_by_name(name_try)
                if find_addr:
                    break

    if find_addr:
        w(f"find_shape_model 地址: 0x{find_addr:X}")
        w(f"函数大小: {get_func_size(find_addr)} bytes")
        w()
        w("```c")
        pseudo = get_func_decompiled(find_addr)
        w(pseudo)
        w("```")
    else:
        w("[未找到 find_shape_model 函数]")
        w("请手动设置 FIND_SHAPE_MODEL_ADDR 为正确的入口地址")
        w()
        w("尝试在当前地址反编译:")
        w("```c")
        pseudo = get_func_decompiled(FIND_SHAPE_MODEL_ADDR)
        w(pseudo)
        w("```")

    w()
    w()

    # ----------------------------------------------------------
    # Part 5: 额外 — 所有已知常量的完整 dump
    # ----------------------------------------------------------
    w("=" * 80)
    w("Part 5: .rdata 常量区完整 dump (0x1800D6A00 - 0x1800D7500)")
    w("=" * 80)
    w()

    base = 0x1800D6A00
    end = 0x1800D7500
    step = 16
    for addr in range(base, end, step):
        data = read_bytes_at(addr, step)
        if data:
            hex_str = " ".join(f"{b:02X}" for b in data)
            # 尝试解析为 float 和 double
            floats = struct.unpack('<ffff', data)
            doubles = struct.unpack('<dd', data)
            float_str = " | ".join(f"{f:.10g}" for f in floats)
            double_str = " | ".join(f"{d:.15g}" for d in doubles)
            w(f"  {addr:016X}: {hex_str}")
            w(f"    float:  {float_str}")
            w(f"    double: {double_str}")
        else:
            w(f"  {addr:016X}: [读取失败]")
    w()

    # ----------------------------------------------------------
    # 写入文件
    # ----------------------------------------------------------
    text = "\n".join(out)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n{'='*60}")
    print(f"导出完成! 输出文件: {OUTPUT_FILE}")
    print(f"总行数: {len(out)}")
    print(f"{'='*60}")


# 运行
if __name__ == "__main__":
    main()
else:
    main()

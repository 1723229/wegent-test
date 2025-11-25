#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的减法计算器
Simple Subtraction Calculator
"""

def subtract(a, b):
    """
    执行减法运算
    Args:
        a (float): 被减数
        b (float): 减数
    Returns:
        float: 差值 (a - b)
    """
    return a - b

def get_number(prompt):
    """
    获取用户输入的数字
    Args:
        prompt (str): 提示信息
    Returns:
        float: 用户输入的数字
    """
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("请输入有效的数字！")

def main():
    """
    主函数 - 运行减法计算器
    """
    print("=" * 30)
    print("     减法计算器")
    print("  Subtraction Calculator")
    print("=" * 30)

    while True:
        try:
            # 获取被减数
            minuend = get_number("请输入被减数: ")

            # 获取减数
            subtrahend = get_number("请输入减数: ")

            # 计算结果
            result = subtract(minuend, subtrahend)

            # 显示结果
            print(f"\n结果: {minuend} - {subtrahend} = {result}")
            print(f"Result: {minuend} - {subtrahend} = {result}\n")

            # 询问是否继续
            continue_calc = input("是否继续计算？(y/n): ").lower().strip()
            if continue_calc not in ['y', 'yes', '是']:
                print("感谢使用减法计算器！")
                break

        except KeyboardInterrupt:
            print("\n\n程序已退出。再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重试...")

if __name__ == "__main__":
    main()
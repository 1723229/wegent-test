#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加法计算器 - 简单的加法运算工具

这个程序提供了一个用户友好的加法计算器界面，
支持多个数字相加并显示计算结果。
"""

def add_numbers(*numbers):
    """
    计算多个数字的和

    参数:
        *numbers: 可变参数，接受任意数量的数字

    返回:
        所有数字的和
    """
    return sum(numbers)

def get_user_input():
    """
    获取用户输入的数字

    返回:
        用户输入的数字列表
    """
    numbers = []
    print("=== 加法计算器 ===")
    print("请输入要相加的数字，输入 'q' 或 'quit' 结束输入")

    while True:
        user_input = input("请输入数字 (或 'q' 结束): ").strip()

        if user_input.lower() in ['q', 'quit']:
            break

        try:
            number = float(user_input)
            numbers.append(number)
            print(f"已添加数字: {number}")
        except ValueError:
            print("请输入有效的数字!")

    return numbers

def main():
    """
    主函数 - 运行加法计算器
    """
    print("欢迎使用加法计算器!")

    while True:
        numbers = get_user_input()

        if not numbers:
            print("没有输入任何数字，程序结束。")
            break

        result = add_numbers(*numbers)

        print(f"\n计算结果:")
        print(f"输入的数字: {numbers}")
        print(f"相加的和: {result}")

        if len(numbers) > 1:
            expression = " + ".join(map(str, numbers))
            print(f"计算过程: {expression} = {result}")

        continue_calc = input("\n是否继续计算? (y/n): ").strip().lower()
        if continue_calc != 'y':
            print("感谢使用加法计算器，再见!")
            break

        print()  # 空行分隔

if __name__ == "__main__":
    main()
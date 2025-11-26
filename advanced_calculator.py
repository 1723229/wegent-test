#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Universal Calculator
支持各种数学运算的综合计算器

功能包括:
- 基础四则运算
- 科学计算(三角函数、对数、指数等)
- 高级数学(阶乘、排列组合、矩阵运算等)
- 单位转换
- 表达式解析和计算
- 统计函数
- 数论函数
"""

import math
import cmath
import re
import numpy as np
import sympy as sp
from typing import Union, List, Dict, Any, Tuple
from fractions import Fraction
from decimal import Decimal, getcontext
import statistics

# 设置高精度计算
getcontext().prec = 50

class UniversalCalculator:
    """万能计算器类"""

    def __init__(self):
        """初始化计算器"""
        self.memory = 0  # 存储器
        self.history = []  # 计算历史
        self.angle_mode = 'rad'  # 角度模式: rad(弧度) 或 deg(角度)

        # 物理常数
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'c': 299792458,  # 光速 m/s
            'g': 9.80665,    # 重力加速度 m/s²
            'h': 6.62607015e-34,  # 普朗克常数 J⋅s
            'k': 1.380649e-23,    # 玻尔兹曼常数 J/K
            'Na': 6.02214076e23,  # 阿伏伽德罗常数
            'R': 8.314462618,     # 气体常数 J/(mol⋅K)
        }

        # 单位转换字典
        self.unit_conversions = {
            'length': {
                'm': 1,
                'km': 1000,
                'cm': 0.01,
                'mm': 0.001,
                'in': 0.0254,
                'ft': 0.3048,
                'yard': 0.9144,
                'mile': 1609.344,
            },
            'weight': {
                'kg': 1,
                'g': 0.001,
                'lb': 0.453592,
                'oz': 0.0283495,
                'ton': 1000,
            },
            'temperature': {
                'celsius': lambda c: c,
                'fahrenheit': lambda f: (f - 32) * 5/9,
                'kelvin': lambda k: k - 273.15,
            },
            'area': {
                'm²': 1,
                'km²': 1000000,
                'cm²': 0.0001,
                'ft²': 0.092903,
                'acre': 4046.86,
            },
            'volume': {
                'l': 1,
                'ml': 0.001,
                'm³': 1000,
                'gallon': 3.78541,
                'cup': 0.236588,
            }
        }

    def basic_operations(self, a: float, b: float, operation: str) -> float:
        """基础四则运算"""
        operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else float('inf'),
            '//': lambda x, y: x // y if y != 0 else float('inf'),
            '%': lambda x, y: x % y if y != 0 else float('nan'),
            '**': lambda x, y: x ** y,
            '^': lambda x, y: x ** y,
        }

        if operation not in operations:
            raise ValueError(f"不支持的运算符: {operation}")

        result = operations[operation](a, b)
        self._add_to_history(f"{a} {operation} {b} = {result}")
        return result

    def scientific_functions(self, func_name: str, x: float, y: float = None) -> float:
        """科学计算函数"""
        # 角度转换
        if self.angle_mode == 'deg' and func_name in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
            if func_name.startswith('a'):  # 反三角函数
                result = getattr(math, func_name)(x)
                return math.degrees(result)
            else:  # 正三角函数
                x = math.radians(x)

        functions = {
            # 三角函数
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': lambda a, b: math.atan2(a, b) if b is not None else None,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'asinh': math.asinh,
            'acosh': math.acosh,
            'atanh': math.atanh,

            # 对数和指数
            'log': math.log10,
            'ln': math.log,
            'log2': math.log2,
            'exp': math.exp,
            'exp2': lambda x: 2 ** x,
            'pow': lambda a, b: a ** b if b is not None else None,

            # 根号和幂
            'sqrt': math.sqrt,
            'cbrt': lambda x: x ** (1/3),
            'square': lambda x: x ** 2,
            'cube': lambda x: x ** 3,

            # 其他
            'abs': abs,
            'floor': math.floor,
            'ceil': math.ceil,
            'round': round,
            'fmod': lambda a, b: math.fmod(a, b) if b is not None else None,
            'gcd': lambda a, b: math.gcd(int(a), int(b)) if b is not None else None,
            'lcm': lambda a, b: abs(int(a) * int(b)) // math.gcd(int(a), int(b)) if b is not None else None,
        }

        if func_name not in functions:
            raise ValueError(f"不支持的函数: {func_name}")

        try:
            if y is not None:
                result = functions[func_name](x, y)
            else:
                result = functions[func_name](x)

            self._add_to_history(f"{func_name}({x}{f', {y}' if y else ''}) = {result}")
            return result
        except Exception as e:
            raise ValueError(f"计算错误: {e}")

    def advanced_math(self, func_name: str, *args) -> Union[float, int, List]:
        """高级数学函数"""
        functions = {
            'factorial': self._factorial,
            'permutation': self._permutation,
            'combination': self._combination,
            'fibonacci': self._fibonacci,
            'prime_factors': self._prime_factors,
            'is_prime': self._is_prime,
            'gamma': math.gamma,
            'beta': self._beta,
            'erf': math.erf,
            'erfc': math.erfc,
        }

        if func_name not in functions:
            raise ValueError(f"不支持的高级数学函数: {func_name}")

        result = functions[func_name](*args)
        self._add_to_history(f"{func_name}({', '.join(map(str, args))}) = {result}")
        return result

    def _factorial(self, n: int) -> int:
        """计算阶乘"""
        if n < 0:
            raise ValueError("阶乘的参数不能为负数")
        return math.factorial(int(n))

    def _permutation(self, n: int, r: int) -> int:
        """计算排列数 P(n,r)"""
        return math.perm(int(n), int(r))

    def _combination(self, n: int, r: int) -> int:
        """计算组合数 C(n,r)"""
        return math.comb(int(n), int(r))

    def _fibonacci(self, n: int) -> int:
        """计算斐波那契数列第n项"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    def _prime_factors(self, n: int) -> List[int]:
        """计算质因数分解"""
        factors = []
        n = int(abs(n))
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    def _is_prime(self, n: int) -> bool:
        """判断是否为质数"""
        n = int(n)
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def _beta(self, x: float, y: float) -> float:
        """Beta函数"""
        return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

    def matrix_operations(self, operation: str, matrix1: List[List], matrix2: List[List] = None) -> List[List]:
        """矩阵运算"""
        m1 = np.array(matrix1)

        if operation in ['add', 'subtract', 'multiply', 'dot'] and matrix2 is not None:
            m2 = np.array(matrix2)

            operations = {
                'add': lambda a, b: a + b,
                'subtract': lambda a, b: a - b,
                'multiply': lambda a, b: a * b,  # 元素wise乘法
                'dot': lambda a, b: np.dot(a, b),  # 矩阵乘法
            }
            result = operations[operation](m1, m2)

        else:
            operations = {
                'transpose': lambda a: np.transpose(a),
                'inverse': lambda a: np.linalg.inv(a),
                'determinant': lambda a: np.linalg.det(a),
                'rank': lambda a: np.linalg.matrix_rank(a),
                'trace': lambda a: np.trace(a),
                'eigenvalues': lambda a: np.linalg.eigvals(a),
                'norm': lambda a: np.linalg.norm(a),
            }

            if operation not in operations:
                raise ValueError(f"不支持的矩阵运算: {operation}")

            result = operations[operation](m1)

        # 转换numpy数组为Python列表
        if isinstance(result, np.ndarray):
            result = result.tolist()

        self._add_to_history(f"矩阵{operation}: {result}")
        return result

    def statistics_functions(self, data: List[float], func_name: str) -> float:
        """统计函数"""
        functions = {
            'mean': statistics.mean,
            'median': statistics.median,
            'mode': statistics.mode,
            'stdev': statistics.stdev,
            'variance': statistics.variance,
            'harmonic_mean': statistics.harmonic_mean,
            'geometric_mean': statistics.geometric_mean,
            'min': min,
            'max': max,
            'range': lambda x: max(x) - min(x),
            'sum': sum,
            'count': len,
        }

        if func_name not in functions:
            raise ValueError(f"不支持的统计函数: {func_name}")

        result = functions[func_name](data)
        self._add_to_history(f"{func_name}({data}) = {result}")
        return result

    def unit_conversion(self, value: float, from_unit: str, to_unit: str, unit_type: str) -> float:
        """单位转换"""
        if unit_type not in self.unit_conversions:
            raise ValueError(f"不支持的单位类型: {unit_type}")

        conversions = self.unit_conversions[unit_type]

        if unit_type == 'temperature':
            # 温度转换特殊处理
            if from_unit == 'celsius':
                celsius = value
            elif from_unit == 'fahrenheit':
                celsius = (value - 32) * 5/9
            elif from_unit == 'kelvin':
                celsius = value - 273.15
            else:
                raise ValueError(f"不支持的温度单位: {from_unit}")

            if to_unit == 'celsius':
                result = celsius
            elif to_unit == 'fahrenheit':
                result = celsius * 9/5 + 32
            elif to_unit == 'kelvin':
                result = celsius + 273.15
            else:
                raise ValueError(f"不支持的温度单位: {to_unit}")
        else:
            # 其他单位转换
            if from_unit not in conversions or to_unit not in conversions:
                raise ValueError(f"不支持的单位: {from_unit} 或 {to_unit}")

            # 转换为基础单位，再转换为目标单位
            base_value = value * conversions[from_unit]
            result = base_value / conversions[to_unit]

        self._add_to_history(f"{value} {from_unit} = {result} {to_unit}")
        return result

    def evaluate_expression(self, expression: str) -> float:
        """解析并计算数学表达式"""
        try:
            # 替换常数
            expr = expression
            for name, value in self.constants.items():
                expr = expr.replace(name, str(value))

            # 使用sympy进行符号计算
            result = float(sp.sympify(expr).evalf())
            self._add_to_history(f"{expression} = {result}")
            return result
        except Exception as e:
            raise ValueError(f"表达式计算错误: {e}")

    def solve_equation(self, equation: str, variable: str = 'x'):
        """解方程"""
        try:
            # 使用sympy解方程
            x = sp.Symbol(variable)
            eq = sp.Eq(sp.sympify(equation.split('=')[0]), sp.sympify(equation.split('=')[1]))
            solutions = sp.solve(eq, x)

            self._add_to_history(f"解方程 {equation}: {solutions}")
            return solutions
        except Exception as e:
            raise ValueError(f"方程求解错误: {e}")

    def derivative(self, expression: str, variable: str = 'x'):
        """求导数"""
        try:
            x = sp.Symbol(variable)
            expr = sp.sympify(expression)
            derivative = sp.diff(expr, x)

            result = str(derivative)
            self._add_to_history(f"d/d{variable}({expression}) = {result}")
            return result
        except Exception as e:
            raise ValueError(f"求导错误: {e}")

    def integral(self, expression: str, variable: str = 'x', limits: Tuple = None):
        """求积分"""
        try:
            x = sp.Symbol(variable)
            expr = sp.sympify(expression)

            if limits:
                # 定积分
                result = sp.integrate(expr, (x, limits[0], limits[1]))
            else:
                # 不定积分
                result = sp.integrate(expr, x)

            result_str = str(result)
            self._add_to_history(f"∫({expression})d{variable} = {result_str}")
            return result_str
        except Exception as e:
            raise ValueError(f"积分计算错误: {e}")

    def complex_operations(self, operation: str, z1: complex, z2: complex = None):
        """复数运算"""
        operations = {
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'multiply': lambda a, b: a * b,
            'divide': lambda a, b: a / b if b != 0 else complex('inf'),
            'power': lambda a, b: a ** b,
            'conjugate': lambda a, b: a.conjugate(),
            'abs': lambda a, b: abs(a),
            'phase': lambda a, b: cmath.phase(a),
            'real': lambda a, b: a.real,
            'imag': lambda a, b: a.imag,
            'polar': lambda a, b: cmath.polar(a),
            'rect': lambda a, b: cmath.rect(a.real, a.imag),
        }

        if operation not in operations:
            raise ValueError(f"不支持的复数运算: {operation}")

        result = operations[operation](z1, z2)
        self._add_to_history(f"复数{operation}: {z1}{f', {z2}' if z2 else ''} = {result}")
        return result

    def number_base_conversion(self, number: str, from_base: int, to_base: int) -> str:
        """进制转换"""
        try:
            # 先转换为十进制
            if from_base == 10:
                decimal_num = int(number)
            else:
                decimal_num = int(number, from_base)

            # 从十进制转换为目标进制
            if to_base == 10:
                result = str(decimal_num)
            elif to_base == 2:
                result = bin(decimal_num)[2:]
            elif to_base == 8:
                result = oct(decimal_num)[2:]
            elif to_base == 16:
                result = hex(decimal_num)[2:].upper()
            else:
                # 通用进制转换
                if decimal_num == 0:
                    result = "0"
                else:
                    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    result = ""
                    num = abs(decimal_num)
                    while num > 0:
                        result = digits[num % to_base] + result
                        num //= to_base
                    if decimal_num < 0:
                        result = "-" + result

            self._add_to_history(f"{number}({from_base}) = {result}({to_base})")
            return result
        except Exception as e:
            raise ValueError(f"进制转换错误: {e}")

    def memory_operations(self, operation: str, value: float = None) -> float:
        """存储器操作"""
        operations = {
            'store': lambda v: setattr(self, 'memory', v) or v,
            'recall': lambda v: self.memory,
            'add': lambda v: setattr(self, 'memory', self.memory + v) or self.memory,
            'subtract': lambda v: setattr(self, 'memory', self.memory - v) or self.memory,
            'clear': lambda v: setattr(self, 'memory', 0) or 0,
        }

        if operation not in operations:
            raise ValueError(f"不支持的存储器操作: {operation}")

        result = operations[operation](value)
        self._add_to_history(f"Memory {operation}: {result}")
        return result

    def set_angle_mode(self, mode: str):
        """设置角度模式"""
        if mode.lower() in ['rad', 'radian', 'radians']:
            self.angle_mode = 'rad'
        elif mode.lower() in ['deg', 'degree', 'degrees']:
            self.angle_mode = 'deg'
        else:
            raise ValueError("角度模式必须是 'rad' 或 'deg'")

        self._add_to_history(f"角度模式设置为: {self.angle_mode}")

    def get_history(self) -> List[str]:
        """获取计算历史"""
        return self.history.copy()

    def clear_history(self):
        """清除历史记录"""
        self.history.clear()
        self._add_to_history("历史记录已清除")

    def _add_to_history(self, record: str):
        """添加记录到历史"""
        self.history.append(record)
        if len(self.history) > 1000:  # 限制历史记录数量
            self.history.pop(0)


def interactive_calculator():
    """交互式计算器界面"""
    calc = UniversalCalculator()

    print("🔢 万能计算器 - 支持各种数学运算")
    print("=" * 50)
    print("输入 'help' 查看帮助，输入 'quit' 退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n计算器 > ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if user_input.lower() == 'help':
                print_help()
                continue

            if user_input.lower() == 'history':
                history = calc.get_history()
                print("\n计算历史:")
                for i, record in enumerate(history[-10:], 1):  # 显示最近10条
                    print(f"{i:2d}. {record}")
                continue

            if user_input.lower() == 'clear':
                calc.clear_history()
                print("历史记录已清除")
                continue

            if user_input.lower() == 'memory':
                print(f"存储器内容: {calc.memory}")
                continue

            if user_input.lower().startswith('mode'):
                parts = user_input.split()
                if len(parts) == 2:
                    calc.set_angle_mode(parts[1])
                    print(f"角度模式已设置为: {calc.angle_mode}")
                else:
                    print("用法: mode [rad|deg]")
                continue

            # 处理各种计算命令
            if user_input.startswith('matrix'):
                handle_matrix_command(calc, user_input)
            elif user_input.startswith('stats'):
                handle_stats_command(calc, user_input)
            elif user_input.startswith('convert'):
                handle_conversion_command(calc, user_input)
            elif user_input.startswith('solve'):
                handle_solve_command(calc, user_input)
            elif user_input.startswith('diff'):
                handle_derivative_command(calc, user_input)
            elif user_input.startswith('integral'):
                handle_integral_command(calc, user_input)
            elif user_input.startswith('base'):
                handle_base_conversion_command(calc, user_input)
            elif '=' in user_input and not user_input.startswith('solve'):
                # 解方程
                result = calc.solve_equation(user_input)
                print(f"解: {result}")
            else:
                # 普通表达式计算
                result = calc.evaluate_expression(user_input)
                print(f"结果: {result}")

        except Exception as e:
            print(f"错误: {e}")


def handle_matrix_command(calc, command):
    """处理矩阵命令"""
    # 简单的矩阵操作示例
    try:
        if 'det' in command:
            matrix = [[1, 2], [3, 4]]  # 示例矩阵
            result = calc.matrix_operations('determinant', matrix)
            print(f"行列式: {result}")
        elif 'inv' in command:
            matrix = [[1, 2], [3, 4]]
            result = calc.matrix_operations('inverse', matrix)
            print(f"逆矩阵: {result}")
        else:
            print("支持的矩阵操作: det, inv, transpose, rank, trace")
    except Exception as e:
        print(f"矩阵操作错误: {e}")


def handle_stats_command(calc, command):
    """处理统计命令"""
    # stats mean 1,2,3,4,5
    try:
        parts = command.split()
        if len(parts) >= 3:
            func_name = parts[1]
            data_str = ' '.join(parts[2:])
            data = [float(x.strip()) for x in data_str.split(',')]
            result = calc.statistics_functions(data, func_name)
            print(f"统计结果: {result}")
        else:
            print("用法: stats [mean|median|stdev|variance] data1,data2,data3,...")
    except Exception as e:
        print(f"统计计算错误: {e}")


def handle_conversion_command(calc, command):
    """处理单位转换命令"""
    # convert 100 m km length
    try:
        parts = command.split()
        if len(parts) == 5:
            value = float(parts[1])
            from_unit = parts[2]
            to_unit = parts[3]
            unit_type = parts[4]
            result = calc.unit_conversion(value, from_unit, to_unit, unit_type)
            print(f"转换结果: {result} {to_unit}")
        else:
            print("用法: convert [值] [源单位] [目标单位] [单位类型]")
            print("支持的单位类型: length, weight, temperature, area, volume")
    except Exception as e:
        print(f"单位转换错误: {e}")


def handle_solve_command(calc, command):
    """处理方程求解命令"""
    # solve x^2 - 4 = 0
    try:
        equation = command[5:].strip()  # 移除 'solve '
        result = calc.solve_equation(equation)
        print(f"方程解: {result}")
    except Exception as e:
        print(f"方程求解错误: {e}")


def handle_derivative_command(calc, command):
    """处理求导命令"""
    # diff x^2 + 2*x + 1
    try:
        expression = command[4:].strip()  # 移除 'diff '
        result = calc.derivative(expression)
        print(f"导数: {result}")
    except Exception as e:
        print(f"求导错误: {e}")


def handle_integral_command(calc, command):
    """处理积分命令"""
    # integral x^2
    try:
        expression = command[8:].strip()  # 移除 'integral '
        result = calc.integral(expression)
        print(f"积分: {result}")
    except Exception as e:
        print(f"积分计算错误: {e}")


def handle_base_conversion_command(calc, command):
    """处理进制转换命令"""
    # base 1010 2 10
    try:
        parts = command.split()
        if len(parts) == 4:
            number = parts[1]
            from_base = int(parts[2])
            to_base = int(parts[3])
            result = calc.number_base_conversion(number, from_base, to_base)
            print(f"进制转换: {result}")
        else:
            print("用法: base [数字] [源进制] [目标进制]")
    except Exception as e:
        print(f"进制转换错误: {e}")


def print_help():
    """打印帮助信息"""
    help_text = """
🔢 万能计算器帮助

基本运算:
- 四则运算: +, -, *, /, //, %, **
- 表达式: 2*3 + 4/2
- 常数: pi, e, c, g, h, k, Na, R

科学计算:
- 三角函数: sin(30), cos(pi/4), tan(45)
- 对数: log(100), ln(e), log2(8)
- 幂和根: sqrt(16), cbrt(27), exp(2)

高级数学:
- 阶乘: factorial(5)
- 排列组合: permutation(5,2), combination(10,3)
- 质因数: prime_factors(12)

矩阵运算:
- matrix det - 计算行列式
- matrix inv - 计算逆矩阵

统计函数:
- stats mean 1,2,3,4,5 - 计算平均值
- stats stdev 1,2,3,4,5 - 计算标准差

单位转换:
- convert 100 m km length - 长度转换
- convert 32 fahrenheit celsius temperature - 温度转换

方程求解:
- solve x^2 - 4 = 0 - 解方程
- diff x^2 + 2*x - 求导
- integral x^2 - 积分

进制转换:
- base 1010 2 10 - 二进制转十进制

其他命令:
- history - 查看计算历史
- clear - 清除历史
- memory - 查看存储器
- mode [rad|deg] - 设置角度模式
- help - 显示帮助
- quit - 退出程序
"""
    print(help_text)


if __name__ == "__main__":
    # 检查依赖库
    try:
        import numpy as np
        import sympy as sp
    except ImportError:
        print("请安装依赖库:")
        print("pip install numpy sympy")
        exit(1)

    interactive_calculator()
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

x: int
y: int
content = []

sko_map = {}


def check_less_zero(arr):
    for i in range(len(arr)):
        if arr[i] <= 0:
            return False
    return True


def draw(f, a, b, c, d, approximation_name):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'ko')
    plt.plot(x, f)
    plt.grid()
    plt.title(approximation_name)
    plt.show()


def draw_all(arr, names):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'ko')
    for i, name in zip(arr,names):
        plt.plot(x,i,linewidth=2, label=name)

    plt.grid()
    plt.title("Графики всех аппроксимирующих функций")
    plt.legend(loc='upper left', fontsize='small')
    plt.show()


def cof_cor(Xs, Ys):
    x_mean = np.mean(Xs)
    y_mean = np.mean(Ys)
    numerator = 0
    denominator_x = 0
    denominator_y = 0
    for x, y in zip(Xs, Ys):
        numerator += (x - x_mean) * (y - y_mean)
        denominator_x += (x - x_mean) ** 2
        denominator_y += (y - y_mean) ** 2
    return numerator / sqrt(denominator_x * denominator_y)


def lin_approx(Xs, Ys):
    A = np.array([[sum(Xs ** 2), sum(Xs)], [sum(Xs), len(Xs)]])
    B = np.array([sum(Xs * Ys), sum(Ys)])
    a, b = np.linalg.solve(A, B)
    return a, b


def evaluate_approximation(R):
    if R >= 0.95:
        print("Высокая точность аппроксимации (модель хорошо описывает явление)")
    elif 0.75 <= R < 0.95:
        print("Удовлетворительная аппроксимация (модель в целом адекватно описывает явление)")
    elif 0.5 <= R < 0.75:
        print("Слабая аппроксимация (модель слабо описывает явление)")
    else:
        print("Точность аппроксимации недостаточна, модель требует изменения")


def print_res(S, delta, R_squared, xs, ys, f, eps):
    print(f'Мера отклонения = {S}')
    print(f'СКО = {delta}')
    print(f"R^2= {R_squared}")
    evaluate_approximation(R_squared)
    print("x_i: ", " , ".join(map(str, xs)))
    print("y_i: ", " , ".join(map(str, ys)))
    print("f(x_i): ", " , ".join(map(str, f)))
    print("eps_i: ", " , ".join(map(str, eps)))
    print()


def linear(array):
    xs = np.array(array[0])
    ys = np.array(array[1])
    name="Линейная аппроксимация"
    # Подсчет аппроксимации
    a, b = lin_approx(xs, ys)
    f = a * xs + b
    eps = ys - f
    # СКО
    delta = (sum(eps ** 2) / len(xs)) ** 0.5
    sko_map['Линейная аппроксимация'] = delta
    # мера отклонения
    S = sum(eps ** 2)
    # Коэффициент детерминации
    y_mean = np.mean(f)
    SS_total = sum((ys - y_mean) ** 2)
    R_squared = 1 - (S / SS_total)
    print(f'\n\nЛинейная аппроксимация:\nf = {a} * x + {b}')
    print(f'Коэффициент корреляции Пирсона r= {cof_cor(xs, ys)}')
    print_res(S, delta, R_squared, xs, ys, f, eps)
    draw(f, a, b, 0, 0, "Линейная аппроксимация")
    return f, name


def squared(array):
    xs = np.array(array[0])
    ys = np.array(array[1])
    name='Квадратичная аппроксимация'
    # Подсчет аппроксимации
    A = np.array([[len(xs), sum(xs), sum(xs ** 2)],
                  [sum(xs), sum(xs ** 2), sum(xs ** 3)],
                  [sum(xs ** 2), sum(xs ** 3), sum(xs ** 4)]])
    B = np.array([sum(ys), sum(xs * ys), sum((xs ** 2) * ys)])
    c, b, a = np.linalg.solve(A, B)
    f = a * xs ** 2 + b * xs + c
    # СКО
    eps = ys - f
    delta = (sum(eps ** 2) / len(xs)) ** 0.5
    sko_map['Квадратичная аппроксимация'] = delta
    # Мера отклонения
    S = sum(eps ** 2)
    # Коэффициент детерминации
    y_mean = np.mean(f)
    SS_total = sum((ys - y_mean) ** 2)
    R_squared = 1 - (S / SS_total)
    print(f'\n\nКвадратичная аппроксимация\nf = {a} * x^2 + {b} * x +{c}')
    print_res(S, delta, R_squared, xs, ys, f, eps)
    draw(f, a, b, c, 0, "Квадратичная аппроксимация")
    return f,name


def triple(array):
    name='Третичная аппроксимация'
    xs = np.array(array[0])
    ys = np.array(array[1])
    # Подсчет аппроксимации
    A = np.array([[len(xs), sum(xs), sum(xs ** 2), sum(xs ** 3)],
                  [sum(xs), sum(xs ** 2), sum(xs ** 3), sum(xs ** 4)],
                  [sum(xs ** 2), sum(xs ** 3), sum(xs ** 4), sum(xs ** 5)],
                  [sum(xs ** 3), sum(xs ** 4), sum(xs ** 5), sum(xs ** 6)]])
    B = np.array([sum(ys), sum(xs * ys), sum((xs ** 2) * ys), sum((xs ** 3) * ys)])
    d, c, b, a = np.linalg.solve(A, B)
    f = a * xs ** 3 + b * xs ** 2 + c * xs + d
    # СКО
    eps = ys - f
    delta = (sum(eps ** 2) / len(xs)) ** 0.5
    sko_map['Третичная аппроксимация'] = delta
    # Мера отклонения
    S = sum(eps ** 2)
    # Коэффициент детерминации
    y_mean = np.mean(f)
    SS_total = sum((ys - y_mean) ** 2)
    R_squared = 1 - (S / SS_total)
    print(f'\n\nТретичная аппроксимация\nf = {a} * x^3 + {b} * x^2 + {c} * x + {d}')
    print_res(S, delta, R_squared, xs, ys, f, eps)
    draw(f, a, b, c, d, "Третичная аппроксимация")
    return f,name


def power(array):
    name='Степенная аппроксимация'
    xs = np.array(array[0])
    ys = np.array(array[1])
    if check_less_zero(xs) and (check_less_zero(ys)):
        log_xs = np.log(xs)
        log_ys = np.log(ys)
    else:
        print("\nНевозможно построить степенную аппроксимацию\n")
        return [],[]

    # Подсчет аппроксимации
    b, a = lin_approx(log_xs, log_ys)
    a = np.exp(a)
    f = a * (xs ** b)
    # СКО
    eps = ys - f
    delta = (sum(eps ** 2) / len(xs)) ** 0.5
    sko_map['Степенная аппроксимация'] = delta
    # Мера отклонения
    S = sum(eps ** 2)
    # Коэффициент детерминации
    y_mean = np.mean(f)
    SS_total = sum((ys - y_mean) ** 2)
    R_squared = 1 - (S / SS_total)
    print(f'\n\nСтепенная аппроксимация\nf = {a} * x ^ {b}')
    print_res(S, delta, R_squared, xs, ys, f, eps)
    draw(f, a, b, 0, 0, "Степенная аппроксимация")
    return f,name


def exponential(array):
    name='Экспоненциальная аппроксимация'
    xs = np.array(array[0])
    ys = np.array(array[1])
    if ( check_less_zero(ys)):
        log_ys = np.log(ys)
    else:
        print("\nНевозможно построить экспонециальную аппроксимацию\n")
        return [],[]
    # Подсчет аппроксимации
    b, a = lin_approx(xs, log_ys)
    a = np.exp(a)
    f = a * (np.exp(xs * b))
    # СКО
    eps = ys - f
    delta = (sum(eps ** 2) / len(xs)) ** 0.5
    sko_map['Экспоненциальная аппроксимация'] = delta
    # Мера отклонения
    S = sum(eps ** 2)
    # Коэффициент детерминации
    y_mean = np.mean(f)
    SS_total = sum((ys - y_mean) ** 2)
    R_squared = 1 - (S / SS_total)
    print(f'\n\nЭкспоненциальная аппроксимация\nf = {a} * e^(x * {b})')
    print_res(S, delta, R_squared, xs, ys, f, eps)
    draw(f, a, b, 0, 0, "Экспоненциальная аппроксимация")
    return f,name


def logarithm(array):
    name='Логарифмическая аппроксимация'
    xs = np.array(array[0])
    ys = np.array(array[1])
    if check_less_zero(xs):
        log_xs = np.log(xs)
    else:
        print("\nНевозможно построить логарифмическую аппроксимацию\n")
        return [],[]

    # Подсчет аппроксимации
    a, b = lin_approx(log_xs, ys)
    f = a * log_xs + b
    # СКО
    eps = ys - f
    delta = (sum(eps ** 2) / len(xs)) ** 0.5
    sko_map['Логарифмическая аппроксимация'] = delta
    # Мера отклонения
    S = sum(eps ** 2)
    y_mean = np.mean(f)
    SS_total = sum((ys - y_mean) ** 2)
    # Коэффициент детерминации
    R_squared = 1 - (S / SS_total)
    print(f'\n\nЛогарифмическая аппроксимация\nf = {a} * ln(x) + {b}')
    print_res(S, delta, R_squared, xs, ys, f, eps)
    print()
    draw(f, a, b, 0, 0, "Логарифмическая аппроксимация")
    return f,name


def main():
    again = True
    while again:
        again = False
        in_type = input('Введите:\n\t* k - если вводить с клавиатуры\n\t* f - если хотите вводить из файла\n')
        if in_type.strip() == 'k':
            print("Введите в одной строке значения x, в другой y")
            line_x = input()
            content.append([float(x) for x in line_x.split(" ")])
            line_y = input()
            content.append([float(x) for x in line_y.split(" ")])
        elif in_type.strip() == 'f':
            file_name = input("Введите имя файла (в первой строке файла должны быть x, во второй y) \n")
            try:
                with open(file_name) as f:
                    for line in f:
                        content.append([float(x) for x in line.split(" ")])
            except FileNotFoundError:
                print("Файла с таким именем нет")
        else:
            print('Введено неверно, попробуйте снова.')
            again = True
    global x
    x = np.array(content[0])
    global y
    y = np.array(content[1])
    if (len(content) <=4):
        linear_f, lin_name = linear(content)
        squared_f, squ_name = squared(content)
        triple_f, trip_name = triple(content)
        power_f, pow_name = power(content)
        exponential_f, exp_name = exponential(content)
        logarithmic_f, log_name = logarithm(content)

        functions_to_draw=[]
        names_function_to_draw=[]
        functions=[linear_f,squared_f,triple_f,power_f,exponential_f,logarithmic_f]
        names=[lin_name,squ_name,trip_name,pow_name,exp_name,log_name]
        for function, function_name in zip(functions, names):
            if len(function)>1:
                functions_to_draw.append(function)
                names_function_to_draw.append(function_name)



        draw_all(functions_to_draw,names_function_to_draw)
        best_approximation = min(sko_map, key=sko_map.get)

        print(f"Лучшая аппроксимация: {best_approximation}, СКО={sko_map[best_approximation]}")
    else:
        print("Введено слишком мало точек, нужно минимум 5")


if __name__ == '__main__':
    main()

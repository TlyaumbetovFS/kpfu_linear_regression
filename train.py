import pandas as pd
import matplotlib.pyplot as plt


def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val


def denormalize_price(normalized_price, min_price, max_price):
    return normalized_price * (max_price - min_price) + min_price


def calculate_mae(y_true, y_pred):
    return (abs(y_pred - y_true)).mean()


def calculate_mse(y_true, y_pred):
    return ((y_pred - y_true) ** 2).mean()


def calculate_r2_score(y_true, y_pred):
    mean_y_true = y_true.mean()
    ss_total = ((y_true - mean_y_true) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    return 1 - (ss_residual / ss_total)


def print_graph(mileage, price, theta0, theta1, min_km, max_km, min_price, max_price):
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, price, alpha=0.7, label='Исходные данные')

    x_line_orig = [min_km, max_km]
    y_line_orig = []

    for x_orig in x_line_orig:
        x_normalized = (x_orig - min_km) / (max_km - min_km)
        y_pred_normalized = theta0 + theta1 * x_normalized
        y_pred_orig = denormalize_price(y_pred_normalized, min_price, max_price)
        y_line_orig.append(y_pred_orig)

    plt.plot(x_line_orig, y_line_orig, color='red', linewidth=2, label='Линия регрессии')

    plt.title('График распределения данных и полученной линейной функции')
    plt.xlabel('Пробег (km)')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    try:
        data = pd.read_csv('data.csv')
        mileage = data['km']
        price = data['price']
    except FileNotFoundError:
        print("Ошибка: файл 'data.csv' не найден!")
        return

    mileage_normalized, min_km, max_km = normalize_data(mileage)
    price_normalized, min_price, max_price = normalize_data(price)

    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.5
    epochs = 1000
    m = len(mileage)

    for epoch in range(epochs):
        sum_error0 = 0
        sum_error1 = 0

        for i in range(m):
            prediction = theta0 + theta1 * mileage_normalized.iloc[i]
            error = prediction - price_normalized.iloc[i]

            sum_error0 += error
            sum_error1 += error * mileage_normalized.iloc[i]

        theta0 -= (learning_rate / m) * sum_error0
        theta1 -= (learning_rate / m) * sum_error1

    print(f"Обучение завершено.")
    print(f"Итоговые параметры: theta0 = {theta0}, theta1 = {theta1}")

    try:
        with open('model_parameters.txt', 'w') as f:
            f.write(f"{theta0},{theta1}\n")
            f.write(f"{min_km},{max_km}\n")
            f.write(f"{min_price},{max_price}\n")
        print("Параметры модели сохранены в 'model_parameters.txt'")
    except IOError:
        print("Ошибка при сохранении файла с параметрами.")

    print_graph(mileage, price, theta0, theta1, min_km, max_km, min_price, max_price)

    predictions_normalized = theta0 + theta1 * mileage_normalized
    predictions_denormalized = denormalize_price(predictions_normalized, min_price, max_price)

    mae = calculate_mae(price, predictions_denormalized)
    mse = calculate_mse(price, predictions_denormalized)
    r2_score = calculate_r2_score(price, predictions_denormalized)

    print("\n--- Оценка качества модели ---")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Коэффициент детерминации (R-squared): {r2_score:.4f}")
    print(f"Использованный learning rate: {learning_rate}.")


if __name__ == '__main__':
    main()

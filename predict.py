def denormalize_price(normalized_price, min_price, max_price):
    return normalized_price * (max_price - min_price) + min_price


def main():
    try:
        with open('model_parameters.txt', 'r') as f:
            lines = f.readlines()
            theta0, theta1 = map(float, lines[0].strip().split(','))
            min_km, max_km = map(float, lines[1].strip().split(','))
            min_price, max_price = map(float, lines[2].strip().split(','))
    except FileNotFoundError:
        print("Файл с параметрами модели не найден. Сначала запустите train.py!")
        print("Предсказанная цена: 0")
        return
    except (IOError, IndexError, ValueError):
        print("Ошибка чтения файла с параметрами модели!")
        return

    while True:
        try:
            mileage_input = input("Пожалуйста, введите пробег автомобиля в км: ")
            mileage = float(mileage_input)
            if mileage < 0:
                print("Пробег не может быть отрицательным! Пожалуйста, введите положительное число!")
                continue
            break
        except ValueError:
            print("Ошибка: введите корректное числовое значение!")

    mileage_normalized = (mileage - min_km) / (max_km - min_km)

    predicted_price_normalized = theta0 + theta1 * mileage_normalized

    predicted_price = denormalize_price(predicted_price_normalized, min_price, max_price)

    predicted_price = max(0, predicted_price)

    if mileage < min_km or mileage > max_km:
        print(
            "\nВнимание: введенный пробег находится за пределами диапазона данных, на которых обучалась модель!"
            " Предсказание может быть неточным!")

    print(f"\nПредполагаемая цена автомобиля: {predicted_price:.2f}")


if __name__ == '__main__':
    main()

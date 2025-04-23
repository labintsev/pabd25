from flask import Flask, render_template, request
import json

app = Flask(__name__)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    # Здесь можно добавить обработку полученных чисел
    # Для примера просто возвращаем их обратно
    data = request.get_json()
    
    print(data['number1'])
    
    return {'status': 'success', 'data': 'Числа успешно обработаны'}

if __name__ == '__main__':
    app.run(debug=True)

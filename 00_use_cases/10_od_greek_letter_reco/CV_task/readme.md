# Задание CV (пользоваться разрешено всем, chatgpt тоже):


## Дано:
- Картинка: `letters.png`.
На ней большое количество разноцветных букв греческого алфавита ("μ", "σ", "π", "λ"), которые могут быть повёрнуты.

- Проект базового рабочего детектора: `./simple_detector/README.md`.
Можно использовать его или написать свой.


## Нужно: 
- Сделать end2end решение по распознаванию букв на выданной картинке и оформить его в виде инференс API, с форматом ответа в виде json рода:
```json
{
  "μ": 10, 
  "σ": 20, 
  "π": 30, 
  "λ": 40
}
```

Пример запроса к API:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@letters.png'
```
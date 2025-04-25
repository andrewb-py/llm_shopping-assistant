import pandas as pd
import logging
import json
import requests


class ShoppingAssistant:
    def __init__(self, csv_path: str, openrouter_api_key: str):
        """
        Инициализация ассистента:
         - Загружается база товаров из CSV.
         - Сохраняется API-ключ для openrouter.
         - Устанавливается системный промпт, задающий высокие требования к точности, структурированности и обоснованиям ответов.
         - Инициализируется история диалога.
        """
        try:
            self.dataset = pd.read_csv(csv_path)
            logging.info("База товаров успешно загружена.")
        except Exception as e:
            logging.error(f"Ошибка загрузки CSV: {e}")
            raise

        self.openrouter_api_key = openrouter_api_key
        self.current_basket = []  # Список id выбранных товаров
        self.conversation_state = {}

        # Системный промпт, задающий контекст и требования к ответам
        self.system_prompt = """
            Ты высококвалифицированный шоппинг-ассистент-стилист.
            Твоя задача – помогать пользователям подбирать гардероб по запросам.
            Отвечай строго структурированно, используя формат JSON там, где это требует запрос, и 
            обосновывай выбор товаров.
            В ответах для подбора или обновления корзины приводи список товаров с их id, 
            наименованием и обоснованием выбора.
            Отвечай строго в соответствии с форматом, требуемом в запросе, если он указан.
            Ни в коем случае, не придумывай информацию, которую пользователь не указывает в явном виде.
            **Выполняй глубокий анализ задачи, разбивая её на шаги и обдумывая каждый шаг.**
        """

        # Инициализируем историю диалога с системным промптом.
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def reset_session(self):
        """
        Сброс сессии: очищает историю диалога, оставляя только системный промпт.
        Эта функция будет использоваться при начале нового диалога.
        """
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        self.current_basket = []
        logging.info("Сессия сброшена.")

    def json_parse(self, s: str):
        """
        Пытается распарсить строку s с помощью json.loads()
        """
        try:
            return json.loads(s)
        except Exception as e:
            logging.error(f"Ошибка при парсинге JSON: {e}")

    def call_llm(self, messages: list, max_tokens: int = 2048) -> str:
        """
        Отправляет сообщения (вместе с сохранённой историей диалога) через API OpenRouter,
        используя модель Qwen QWQ 32B. Новые сообщения добавляются в историю диалога,
        а ответ сохраняется для поддержания контекста.
        """
        # Добавляем новые сообщения в историю
        self.conversation_history.extend(messages)

        # Формируем "чистую" историю, где каждое сообщение – словарь с полями "role" и "content" (строка)
        clean_history = []
        for msg in self.conversation_history:
            content = msg.get("content")
            if not isinstance(content, str):
                content = str(content)
            clean_history.append({"role": msg.get("role"), "content": content})

        data = {
            "messages": clean_history,
            "model": "qwen/qwq-32b:free",
            "max_tokens": max_tokens,
            "temperature": 0.5,
            "top_p": 0.9
        }
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result_json = response.json()
            result_msg = result_json.get("choices", [{}])[0].get("message", "").get("content", "")
            if not isinstance(result_msg, str):
                result_msg = str(result_msg)
            logging.info("LLM вернула ответ.")
            self.conversation_history.append({"role": "assistant", "content": result_msg})
            return result_msg
        except Exception as e:
            logging.error(f"Ошибка вызова LLM: {e}")
            return ""

    def parse_query(self, user_message: str) -> dict:
        """
        Извлекает параметры для подбора одежды из запроса пользователя.
        Промпт просит вернуть JSON в формате:
        {
          "Цена": <число или null>,
          "Материалы": <список или null>,
          "Сезонность": <список или null>,
          "Цвет": <список или null>,
          "Стиль": <строка или null>,
          "Личные предпочтения": <строка или null>
        }
        """
        prompt = f"""
            Проанализируй запрос пользователя и выбери наиболее подходящие параметры для дальнейшего подбора одежды. Выбирай параметры исходя из контекста запроса пользователя, но не придумывай их из ниоткуда. Лучше не выбрать параметр и указать его null, чем придумать параметр, который не был указан в запросе и сильно сокращает рассматриваемую базу товаров.
            Определи следующие параметры (в скобках указаны пояснения):\n
            Цена (максимальная цена, число);\n
            Материалы (список материалов, выбирай из: хлопок, деним с эластаном, вискоза,
            полиэстер, смесовая ткань, кожа, шерсть, костюмная ткань, деним, нейлон, замша,
            акрил, резина, синтетический наполнитель, шифон, экокожа, лен, пуховой наполнитель,
            утеплитель, эластан, сетка, металл, синтетика, тюль);\n
            Сезонность (список, выбирай из: лето, весна, осень, зима);\n
            Цвет (список, выбирай из: белый, темно-синие, пастельно-розовое, голубая, черная,
            темно-серая, белые, серое, красный, светло-синие, синий, оливковый, коричневые,
            бежевый, кремовый, темно-фиолетовая, коричневая, синяя, желтое, коричневый,
            оливковые, бежевые, светло-голубое, золотые, розовое, темно-серый, малиновая,
            оливковая, темно-серые, темно-коричневая, светло-серая, изумрудное, мятное,
            многоцветное, светло-голубая, белая, темно-синий, розовая, серебряные, золотой,
            голубые, желтая, синие, серебряная, серые, желтый, зеленый, разноцветные, розовый,
            камель, красная, оранжевое, темно-зеленый, бордовый);\n
            Стиль (один из: повседневный, деловой, романтический, спортивный, вечерний);\n
            Иные предпочтения (пожелания пользователя, строка);\n
            Верни строго JSON в указанном формате. Названия параметров пиши с большой буквы: 'Цвет', 'Стиль', 'Сезонность', 'Цена', 'Материалы', 'Иные предпочтения'.
            Если параметр не указан в контексте запроса пользователя, присвой ему null (в том числе и вместо пустого списка).
            Ни в коем случае не придумывай параметры "Сезонность", "Цена", "Материалы", "Цвет", "Иные предпочтения" если пользователь не указывает на них явно.
            К параметру "Стиль" можно относиться менее строго и добавлять в список несколько 
            значений, чтобы не сужать слишком сильно множество товаров. Например, если это свидание, то можно выбрать [романтический, вечерний], если вечеринка, то [вечерний, повседневный] и так далее.
            Думай последовательно, шаг за шагом анализируя запрос, не придумывай ничего, что не указано в запросе и действуй строго в заданном формате.
            Итак, текущий запрос, по которому надо выбрать параметры: {user_message}\n
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.call_llm(messages)
        params = self.json_parse(response)
        if params is None:
            logging.error("Ошибка парсинга параметров: возвращаем значения по умолчанию")
            params = {
                "Цена": None,
                "Материалы": None,
                "Сезонность": None,
                "Цвет": None,
                "Стиль": None,
                "Личные предпочтения": None
            }
        else:
            logging.info("Параметры запроса успешно извлечены.")
        return params

    def filter_products(self, params: dict) -> pd.DataFrame:
        """
        Фильтрует DataFrame с товарами по параметрам, извлечённым из запроса.
        Для каждого критерия проверяет, задан ли он (не равен None) и отбирает строки, где хотя бы одно из значений (учитывая, что в ячейке может быть несколько признаков через запятую)
        совпадает с указанным параметром.
        """
        filtered = self.dataset.copy()

        # Фильтрация по цене
        if params.get("Цена") is not None:
            try:
                price = float(params["Цена"])
                filtered = filtered[filtered["Цена (рубли)"] <= price]
                logging.info(f"Отфильтровано по цене: {price} рублей.")
            except Exception as e:
                logging.error(f"Ошибка при фильтрации по цене: {e}")

        # Фильтрация по материалам
        if params.get("Материалы"):
            materials = [m.strip().lower() for m in params["Материалы"] if m.strip()]
            filtered = filtered[filtered["Материалы"].apply(
                lambda x: any(mat in x.lower() for mat in materials)
            )]
            logging.info(f"Отфильтровано по материалам: {materials}.")

        # Фильтрация по сезонности
        if params.get("Сезонность"):
            seasons = [s.strip().lower() for s in params["Сезонность"] if s.strip()]
            filtered = filtered[filtered["Сезонность"].apply(
                lambda x: any(season in x.lower() for season in seasons)
            )]
            logging.info(f"Отфильтровано по сезонности: {seasons}.")

        # Фильтрация по цвету
        if params.get("Цвет"):
            colors = [c.strip().lower() for c in params["Цвет"] if c.strip()]
            filtered = filtered[filtered["Цвет"].apply(
                lambda x: any(color in x.lower() for color in colors)
            )]
            logging.info(f"Отфильтровано по цвету: {colors}.")

        # Фильтрация по стилю
        if params.get("Стиль"):
            style_param = params["Стиль"]
            if isinstance(style_param, list):
                styles = [s.strip().lower() for s in style_param if s.strip()]
                filtered = filtered[
                    filtered["Стиль"].apply(lambda x: any(s in x.lower() for s in styles))]
                logging.info(f"Отфильтровано по стилям: {styles}.")
            else:
                style = str(style_param).strip().lower()
                filtered = filtered[filtered["Стиль"].str.lower().str.contains(style, na=False)]
                logging.info(f"Отфильтровано по стилю: {style}.")

        logging.info(f"Осталось товаров после фильтрации: {len(filtered)}.")
        return filtered

    def select_products(self, filtered_df: pd.DataFrame, params: dict) -> dict:
        """
        Выбирает оптимальные товары из отфильтрованного списка.
        Промпт для LLM:
          - Передаётся список товаров (в виде списка словарей).
          - Указываются все параметры, включая личные предпочтения (если заданы).
          - Модель должна выбрать товары, максимально соответствующие заданным критериям, и обосновать свой выбор.
          - Формат ответа: JSON с ключом "selected_products", где каждый элемент – объект с полями "id", "наименование" и "обоснование".
        """
        products_info = filtered_df.to_dict(orient="records")
        prompt = (f"У пользователя заданы следующие параметры: {params}.\n" +
                  """Из следующего списка товаров выбери оптимальные варианты, учитывая все критерии и личные предпочтения.
                  Если пользователь просил составить ему весь образ, то старайся выбрать полный комплект вещей из списка товаров, так чтобы они гармонировали и сочетались между собой.
                  Обрати внимание, что если указан параметр "Цена", то нужно постараться, чтобы товары в сумме не превысили заданный бюджет.
                  Для каждого выбранного товара укажи его id, наименование и обоснование, почему он был выбран.
                  Верни результат в формате JSON следующим образом:
                  {"selected_products": [ {"id": <id>, "наименование": <наименование>, "обоснование": <обоснование>}, ... ]}.\n""" +
                  f"Список товаров: {products_info}")

        messages = [{"role": "user", "content": prompt}]
        response = self.call_llm(messages)
        selection = self.json_parse(response)
        if selection is None:
            logging.error("Ошибка при выборе товаров: возвращаем значения по умолчанию")
            selection = {"selected_products": []}
        else:
            logging.info("Товары успешно выбраны LLM.")
        return selection

    def update_selection(self, user_feedback: str) -> dict:
        """
        Обновляет текущую корзину с учётом обратной связи пользователя.
        Промпт для LLM:
          - Передаётся текущая корзина и обратная связь.
          - Модель должна пересобрать корзину, обосновав изменения.
          - Формат ответа аналогичен select_products.
        """
        prompt = f"""
            Текущая корзина товаров: {self.current_basket}.
            Обратная связь пользователя: {user_feedback}.
            Пересобери корзину с учётом этой обратной связи, выбери оптимальные товары и обоснуй свой выбор.
            Верни результат в формате JSON: {"selected_products": [ {"id": <id>, "наименование": <наименование>, "обоснование": <обоснование>}, ... ]}.'
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.call_llm(messages)
        updated = self.json_parse(response)
        if updated is None:
            logging.error("Ошибка обновления корзины: возвращаем текущую корзину")
            updated = {"selected_products": self.current_basket}
        else:
            # Обновляем текущую корзину, сохраняя список id выбранных товаров
            self.current_basket = [prod.get("id") for prod in updated.get("selected_products", [])]
            logging.info("Корзина успешно обновлена.")
        return updated

    def handle_general_message(self, user_message: str) -> str:
        """
        Отвечает на общее сообщение пользователя в дружественном и профессиональном тоне.
        """
        prompt = (
            f"Пользователь сказал: '{user_message}'. "
            "Ответь дружелюбно и профессионально, предложи помощь, если это уместно." \
            "Если сообщение пользователя не связано с подбором одежды, то вежливо ответь, что ты " \
            "ассистент-стилист и можешь давать советы только по этой теме."
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.call_llm(messages)
        logging.info("Общий ответ сформирован LLM.")
        return response

    def process_user_message(self, user_message: str) -> str:
        """
        Основной метод обработки входящего сообщения.
          1. Определяет тип сообщения: 'новый запрос', 'пересбор корзины' или 'обычное сообщение'.
          2. Для нового запроса извлекает параметры, фильтрует товары и собирает корзину с обоснованием.
          3. Для пересборки корзины обновляет подбор с учётом обратной связи.
          4. Для обычного сообщения отвечает дружественно.
        """
        prompt = f"""
            Определи тип следующего сообщения.
            Если сообщение подразумевает запрос на подбор одежды с нуля, ответь: "новый запрос". 
            Если сообщение содержит корректировку уже сформированной подборки, ответь: "пересбор корзины".
            Если сообщение не связано с подбором одежды, ответь: "обычное сообщение".
            Пример ответа на сообщение "Привет! Я парень, подбери мне образ на вечернее свидание с девушкой":
            "новый запрос"
            Итак, сообщение, тип которого надо определить: "{user_message}"
            """

        messages = [{"role": "user", "content": prompt}]
        response = self.call_llm(messages).lower()
        logging.info(f"Результат анализа сообщения: {response}")

        if "новый запрос" in response:
            # Извлекаем параметры запроса
            params = self.parse_query(user_message)
            # Применяем фильтрацию по параметрам
            filtered = self.filter_products(params)
            if filtered.empty:
                # Если после фильтрации нет товаров, информируем пользователя
                return "К сожалению, по заданным параметрам не найдено товаров. Попробуйте изменить условия запроса."

            # Выбираем товары из отфильтрованного списка
            selection = self.select_products(filtered, params)
            if not selection.get("selected_products"):
                # Если выбор товаров вернул пустой список
                return "К сожалению, не удалось подобрать товары, удовлетворяющие вашим условиям. Попробуйте изменить параметры запроса."

            # Сохраняем текущую корзину как список id выбранных товаров
            self.current_basket = [prod.get("id") for prod in
                                   selection.get("selected_products", [])]
            # Формируем ответ с перечислением товаров и обоснованиями
            answer = "Были выбраны следующие товары:\n"
            for prod in selection.get("selected_products", []):
                answer += f"ID: {prod.get('id')}, Наименование: {prod.get('наименование')}. Обоснование: {prod.get('обоснование')}\n"
            return answer

        elif "пересбор корзины" in response:
            updated = self.update_selection(user_message)
            if not updated.get("selected_products"):
                return "К сожалению, пересбор корзины не дал результатов. Попробуйте изменить условия запроса."
            answer = "Обновленная корзина товаров:\n"
            for prod in updated.get("selected_products", []):
                answer += f"ID: {prod.get('id')}, Наименование: {prod.get('наименование')}. Обоснование: {prod.get('обоснование')}\n"
            return answer

        elif "обычное сообщение" in response:
            return self.handle_general_message(user_message)

        else:
            return self.handle_general_message(user_message)


def main():
    # TODO API подтягивать из переменных среды через os
    csv_path = "data/db.csv"
    api_key = "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"

    assistant = ShoppingAssistant(csv_path, api_key)

    print("Добро пожаловать в Shopping Assistant! Для выхода введите 'exit'.")
    print("Для начала нового диалога введите команду 'reset'.\n")

    while True:
        user_message = input("Введите сообщение: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            print("Диалог завершен.")
            break
        if user_message.lower() == "reset":
            assistant.reset_session()
            print("Сессия сброшена. Начинаем новый диалог.\n")
            continue
        response = assistant.process_user_message(user_message)
        print("Ответ ассистента:\n", response, "\n")


if __name__ == "__main__":
    main()
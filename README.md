# intern-test

Краткая инструкция по запуску:

1. Находясь в корневой папке проекта (где находится Dockerfile), 
запустить сборку образа контейнера командой в терминале Bash: 
docker build -t myimage .
(тег myimage - имя образа)

2. После успешной сборки образа запустить в работу контейнер командой:
docker run -d --name mycontainer -p 8000:8000 myimage
(Параметр -d указывает, что вы запускаете контейнер в отключенном режиме в фоне. 
Значение -p создает сопоставление между портом узла 8000 и портом контейнера 8000. 
Без сопоставления портов вы не сможете получить доступ к приложению)

3. Если у вас установлен плагин для работы с Docker в VS Code, то в области 
Docker в разделе КОНТЕЙНЕРЫ щелкните правой кнопкой мыши myimage и выберите 
команду Открыть в браузере. Или перейдите в веб-браузере по адресу: 
http://localhost:8000

4. Для проверки работы скрипта, перейти по адресу http://localhost:8000/docs
Далее нажав на кнопку "Try it out", в поле ввести тестовый запрос: 
{"text": "Это API сервис"}, нажать снизу кнопку "Execute"
Далее если код выполнился успешно ниже в блоке Responses вы получите
статус код 200 от сервера и ответ в виде списка значений, соответствующих
интонации тестовой фразы, вычисленных с помощью подключенной модели rubert-tiny2.


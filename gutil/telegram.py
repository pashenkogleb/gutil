import requests


def send_msg(msg,token, chat_id = 226309556):
    '''
    chat_id is of my chat with bot
    token is the token of the bot
    '''

    bot_chatID = str(chat_id)
    send_text = 'https://api.telegram.org/bot' + token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + msg
    response = requests.get(send_text)
    assert response.status_code ==200
    res = response.json()
    assert res['ok']
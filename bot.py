import os
import sys
import telebot
import logging
from dotenv import load_dotenv
from model import get_models, predict_game, is_player
from requests.exceptions import ConnectionError, ReadTimeout

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID').split(',')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
models, le, scaler = get_models()


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    msj = message.text
    if ' v ' in msj:
        home, away = msj.split(' v ')
        home = home.strip()
        away = away.strip()
        if is_player(home) and is_player(away):
            print(f'{home} v {away}')
            results = predict_game(
                models, le, scaler, {
                    'home_player': home,
                    'away_player': away,
                })
            model_results = {
                'H': 0,
                'D': 0,
                'A': 0
            }
            for result in results:
                model_results[result] += 1

            model_results['PH'] = (model_results['H']*100/8)
            model_results['PA'] = (model_results['A']*100/8)
            model_results['PD'] = (model_results['D']*100/8)

            result = 'Sin apuesta'
            if model_results['PH'] >= 75:
                result = 'H'
            elif model_results['PA'] >= 75:
                result = 'A'
            elif model_results['PD'] >= 75:
                result = 'D'
            txt_result = f'{home} v {away} -> {result} | {model_results["PH"]}% {model_results["PD"]}% {model_results["PA"]}%' # noqa
            bot.reply_to(message, txt_result)
        else:
            jugador = []
            if not is_player(home):
                jugador.append(home)
            if not is_player(away):
                jugador.append(away)
            bot.reply_to(message, f'Jugador: {jugador} no juega.')
    else:
        bot.reply_to(message, msj)


def start_bot():
    global bot
    logging.info('Ultron BOT')
    try:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except (ConnectionError, ReadTimeout):
        sys.stdout.flush()
        os.execv(sys.argv[0], sys.argv)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Fin...")
        bot.stop_polling()
    else:
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    finally:
        try:
            bot.stop_polling()
        except Exception:
            pass


if __name__ == "__main__":
    start_bot()

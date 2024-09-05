import os
import sys
import telebot
import logging
from collections import Counter
from dotenv import load_dotenv
from model import get_models, predict_game, is_player
from players import get_player_stats
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
models, model_data, le, scaler = get_models()
player_stats = get_player_stats()


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    global player_stats
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

            home_stats = player_stats[home]
            away_stats = player_stats[away]
            home_no = home_stats['No']
            home_wins = home_stats['Ganados']
            home_draws = home_stats['Empate']
            home_loss = home_stats['Perdidos']
            away_no = away_stats['No']
            away_wins = away_stats['Ganados']
            away_draws = away_stats['Empate']
            away_loss = away_stats['Perdidos']

            cara_cara = model_data.query(f"home_player == '{home}' and away_player == '{away}'") # noqa
            result_mapping = {0: 'H', 1: 'D', 2: 'A'}
            cara_cara.loc[:, 'result'] = cara_cara['result'].map(result_mapping) # noqa
            resultados_cara = cara_cara['result'].tolist()
            count = Counter(resultados_cara)

            # print(cara_cara[['home_player', 'away_player', 'home_score', 'away_score', 'result']]) # noqa

            txt_result = f'{home} v {away} -> {result} | {model_results["PH"]}% {model_results["PD"]}% {model_results["PA"]}%\n' # noqa
            txt_result = txt_result + f'{", ".join(resultados_cara[:15])} \n'
            txt_result = txt_result + f"H:{count['H'] if 'H' in count else 0} D:{count['D'] if 'D' in count else 0} A:{count['A'] if 'A' in count else 0}\n\n" # noqa
            txt_result = txt_result + f'{home} [{home_no}] W:{home_wins} D:{home_draws} L:{home_loss}\n\n' # noqa
            txt_result = txt_result + f'{away} [{away_no}] W:{away_wins} D:{away_draws} L:{away_loss}\n\n' # noqa
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

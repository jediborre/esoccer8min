file_name = 'esoccer8min_totalcorner.txt'
base_url = 'https://www.totalcorner.com/league/view/12995/end/Esoccer-Battle-8-mins-play/page:'
urls = [f'{base_url}{i}' for i in range(1, 278)]

# Save the list of URLs to a file
with open(file_name, 'w') as file:
    for url in urls:
        file.write(url + '\n')

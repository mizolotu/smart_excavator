import requests, time

if __name__ == '__main__':

    url = 'http://127.0.0.1:5000/mode'
    modes = ['USER', 'AI']
    DELAY = 1
    mode = ''
    try:
        while True:
            new_mode = ''
            while new_mode not in modes:
                mode = input("Enter command: USER (user control) or AI (switch to AI): ")
                data = {'mode': ''}
                if mode == 'USER':
                    data['mode'] = 'USER'
                elif mode == 'AI':
                    data['mode'] = 'AI_TEST'
                try:
                    r = requests.post(url, json=data)
                    jdata = r.json()
                    new_mode = jdata['mode']
                    dig = jdata['dig_a']
                    if new_mode.startswith('AI'):
                        if dig is not None:
                            print('AI will dig here: {0}'.format(dig))
                        else:
                            print('Sorry, dig operation has not been detected. Try again!')
                    elif new_mode == 'USER':
                        print('Excavator is under user control!')
                    mode = new_mode
                except Exception as e:
                    print(e)
                time.sleep(DELAY)
    except KeyboardInterrupt:
        print('Exiting...')

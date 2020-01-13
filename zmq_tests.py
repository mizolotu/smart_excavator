import zmq, json
from threading import Thread
from time import sleep

def start_server():
    try:
        context = zmq.Context(1)
        frontend = context.socket(zmq.SUB)
        frontend.bind("tcp://*:{0}".format(sub_port))
        frontend.setsockopt_string(zmq.SUBSCRIBE, "")
        backend = context.socket(zmq.PUB)
        backend.bind("tcp://*:{0}".format(pub_port))
        zmq.device(zmq.FORWARDER, frontend, backend)
    except Exception as e:
        print(e)
    finally:
        frontend.close()
        backend.close()
        context.term()

def pub_to_server(id):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect("tcp://localhost:{0}".format(sub_port))
    print('connected')
    while True:
        #socket.send_string('ids {0}'.format(json.dumps({'id': id})))
        socket.send_string('ids {0}'.format(id))
        print('sent')
        sleep(1)

def sub_on_server():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:{0}".format(pub_port))
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    for update_nbr in range(10):
        print(update_nbr)
        string = socket.recv_json()
        print(string)


if __name__ == '__main__':
    sub_port = 5555
    pub_port = 5556
    id = '123'
    th = Thread(target=start_server)
    th.setDaemon(True)
    #th.start()
    print('sleeping')
    sleep(1)
    th = Thread(target=pub_to_server, args=(id,))
    #th.setDaemon(True)
    th.start()
    #print('sleeping')
    #sleep(1)
    #sub_on_server()

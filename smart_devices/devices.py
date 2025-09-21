import tinytuya

try:
# hex
    bulb = tinytuya.BulbDevice(
        dev_id='bfd9d492e56e71e905ebih',
        address=tinytuya.find_device('bfd9d492e56e71e905ebih')['ip'],     # Or set to 'Auto' to auto-discover IP address
        local_key='g=y8~V:7x0zq1q!m',
        version=3.3)
    # bulb.set_socketPersistent(True)
    print(bulb.status())
except:pass
def change_color_of_LED(red,green,blue):
    try:
        bulb._send_receive(
            bulb.generate_payload(
                7,
                {
                    bulb.DPS_INDEX_MODE[bulb.bulb_type]: bulb.DPS_MODE_COLOUR,
                    bulb.DPS_INDEX_COLOUR[bulb.bulb_type]: tinytuya.BulbDevice._rgb_to_hexvalue(red,green,blue, 'B'),
                },
            ),getresponse=False)
    except:pass
    return "done"


# bulb
try:
    d = tinytuya.BulbDevice(
        dev_id='bf03f976dbc55901f9nblx',
        address=tinytuya.find_device('bf03f976dbc55901f9nblx')['ip'],     # Or set to 'Auto' to auto-discover IP address
        local_key="lwJ<Q3[tL)]XlNt2",
        version=3.5)
    print(d.status())
except:pass
# d.set_socketPersistent(True)
def change_color_of_bulb(red,green,blue):
    # d.set_colour(0, 0, 0,True)
    try:
        d.set_colour(red, green, blue,True)
    except:pass

    return 'done'




# change_color_of_bulb(0,255,0)
#
# def change_colorful():
#
#     d._send_receive(
#         d.generate_payload(
#             7,
#             {'51': 'ARcDXl5gAABkADgvAB5cANVFARpk'},
#         )
#     )
#
# d.set_colour(255,0,0,nowait=True)
#
#
#
#
# # plug
# d = tinytuya.OutletDevice(
#     dev_id='bf06e943faa47dbf26gcek',
#     address='192.168.1.22',      # Or set to 'Auto' to auto-discover IP address
#     local_key='-?3-eyTfe:Z`a}jY',
#     version=3.5)
# print(d.status())
#
#
#
# cloud = tinytuya.Cloud(** {'apiKey': '5rwhuqaxvyfp7qc5nf34', 'apiSecret': '336fff88ea2a4ca89ebe3fb2c949de57',
#                          'apiRegion': 'eu', 'apiDeviceID': 'bf489dd05c973d89cbwpur'})
# cloud.getdevices(  include_map=False )
# cloud.getdevices_raw
# ### get temp and hum of both devices
#
#
#
#
# import tinytuya
#
# d = tinytuya.Device('DEVICEID', 'DEVICEIP', 'DEVICEKEY', version=3.3, persist=True)
#
# print(" > Send Request for Status < ")
# d.status(nowait=True)
#
# print(" > Begin Monitor Loop <")
# while(True):
#     # See if any data is available
#     data = d.receive()
#     print('Received Payload: %r' % data)
#
#     # Send keep-alive heartbeat
#     if not data:
#         print(" > Send Heartbeat Ping < ")
#         d.heartbeat()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # bulb
#
#
# import threading
# import tinytuya
#
# d = tinytuya.OutletDevice(
#     dev_id='bf489dd05c973d89cbwpur',
#     address='192.168.1.24',      # Or set to 'Auto' to auto-discover IP address
#     local_key="&S:^JG6IKfz7&;)O",
#     version="")
# print(d.status())
#
#
#
#
#
#
#
# import threading
# import tinytuya
#
# def check_device(a,b,c,d,e):
#     # dev = tinytuya.OutletDevice(
#     dev = tinytuya.Device(
#         a,b,c
#     )
#     dev.set_version(d)
#
#
#     dev.set_dpsUsed({f"{e}": None})
#     print(f'==={dev.status()}')
#
# threading.Thread(target=check_device('bf06e943faa47dbf26gcek','192.168.1.22','-?3-eyTfe:Z`a}jY',3.5,20)).start()
#
#
# threading.Thread(target=check_device('bf06e943faa47dbf26gcek','192.168.1.22','-?3-eyTfe:Z`a}jY',3.5,20)).start()
#
#
#
#
#
#
#
#
#
#
#
#
#
# d = tinytuya.OutletDevice(
#     'bfd9d492e56e71e905ebih',
#     '192.168.1.20',      # Or set to 'Auto' to auto-discover IP address
#     'g=y8~V:7x0zq1q!m',
#     'device22')
# d.set_version(3.3)
# d.set_dpsUsed({"20": None})
# data = d.status()
#
#
#
#
#
#
#
#
#
#
#
# d = tinytuya.OutletDevice(
#     dev_id='bfd9d492e56e71e905ebih',
#     address='192.168.1.20',      # Or set to 'Auto' to auto-discover IP address
#     local_key='g=y8~V:7x0zq1q!m',
#     version=3.3)
# data = d.status()
#
#
#
# # Connect to Device
# d = tinytuya.deviceScan()
# d = tinytuya.find_device('bfd9d492e56e71e905ebih')

# Get Status

# functions = [
#     {
#         'name': 'change_color_bulb',
#         'description': 'change the color of rgb bulb to given rgb values',
#         'parameters': {
#             'name': 'one_or_all',
#             'type': 'string',
#             'description': 'number of lights to turn off or turn off all of them',
#             'required': True
#         }
#     },
# ]


functions = [
    {
        'name': 'change_color_of_bulb',
        'description': 'change the color of rgb bulb to given rgb values',
        'parameters':{
            'type': 'object',
            'properties': {
                'red': {
                    'type': 'int',
                    'description': 'red value in range [0, 255]',
                },
                'green': {
                    'type': 'int',
                    'description': 'green value in range [0, 255]',
                },
                'blue': {
                    'type': 'int',
                    'description': 'blue value in range [0, 255]',
                }
            },
            'required': ['red','green','blue'],
        },
    },
    {
        'name': 'change_color_of_LED',
        'description': 'change the color of rgb LED to given rgb values',
        'parameters':{
            'type': 'object',
            'properties': {
                'red': {
                    'type': 'int',
                    'description': 'red value in range [0, 255]',
                },
                'green': {
                    'type': 'int',
                    'description': 'green value in range [0, 255]',
                },
                'blue': {
                    'type': 'int',
                    'description': 'blue value in range [0, 255]',
                }
            },
            'required': ['red','green','blue'],
        },
    },
]


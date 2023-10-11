import os

wifiName = 'BSS Sleep-Detection'
wifiPassword = 'testing123'

def CreateWifiConfig(SSID, password):
	config_lines = [
		'ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev',
		'update_config=1',
		'country=US',
		'\n',
		'network={',
		'\tssid="{}"'.format(SSID),
		'\tpsk="{}"'.format(password),
		'}'
	]

	config = '\n'.join(config_lines)

	print(config)

	os.popen("sudo chmod a+w /etc/wpa_supplicant/wpa_supplicant.conf")

	with open("/etc/wpa_supplicant/wpa_supplicant.conf", "w") as wifi:
		wifi.write(config)
	
	print("Wifi config added")


CreateWifiConfig(wifiName,wifiPassword)
os.popen("sudo wpa_cli -i wlan0 reconfigure")

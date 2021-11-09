# minigo
This project teaches you how to build an AlphaGo Zero implementation called Minigo, and run it on the Edge TPU

Project summary

The AlphaGo Zero algorithm from DeepMind shocked the AI community when it became the first AI to beat a professional go player, using a relatively simple approach. At its heart, AlphaGo Zero is a convolutional neural network (CNN) that parses the game board using an input tensor similar to an image bitmap. Minigo is a different implementation of the design in the AlphaGo Zero papers, and it uses only open-source tools and libraries.

The fact that the Minigo model uses an architecture similar to other image processing models makes it an excellent fit for the Edge TPU, allowing you to run your own version of the go-playing AI and play it yourself.

How it works

Minigo's model is given a go board as input, and the model provides the answer to two questions: Who is winning, and what are probably good moves to consider? To determine these answers, the input go board is treated like a 19x19 image with 17 "channels" (instead of the 3 RGB channels found in an ordinary image). Each pixel in the 19x19 matrix represents each position of the board, and the 17 channels represent the board in the last 8 moves and whose turn it is. The input is given to a stack of "convolutional blocks," which are convolutional layers with batch-normalization and activations, and joined with residual connections. A network is described by the number of these blocks and their "width," or number of convolutional filters.

1. Install the Minigo model and software

2. Run the code, displaying the game on a monitor

3. Learn how to retrain the Minigo model (optional)

What you'll need

All you need is either a Coral USB Accelerator (connected to a host Linux computer such as a Raspberry Pi with a keyboard/mouse and monitor) or a Coral Dev Board (with a connected monitor and accessible from a computer over SSH/MDT).

Here i am using Coral Dev Board

Get Started

Step 1: Set up your Coral device

If you haven't yet set up your Dev Board, get it connected by following the appropriate Get Started guide(https://coral.ai/docs/dev-board/get-started/), then come back here.

You should have the following software installed in your Coral dev Board

1.	Python3
2.	Docker
3.	Cloud SDK
4.	Virtualenv/virtualenvwrapper

///Install Docker

Install using the repository

Before you install Docker Engine for the first time on a new host machine, you need to set up the Docker repository. Afterward, you can install and update Docker from the repository.
 
Set up the repository

Update the apt package index and install packages to allow apt to use a repository over HTTPS:

	$ sudo apt-get updat
	$ sudo apt-get install \ ca-certificates \ curl \ gnupg \ lsb-release
	
Add Docker’s official GPG key:

	$  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
	$  echo \  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $ (lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	
Install Docker Engine
	
Update the apt package index, and install the latest version of Docker Engine and containerd, or go to the next step to install a specific version

	 $ sudo apt-get update
 	 $ sudo apt-get install docker-ce docker-ce-cli containerd.io
	 
Verify that Docker Engine is installed correctly by running the hello-world image.

	$  sudo docker run hello-world
	
Now Docker is installed successfully

Install Python3

	 $ sudo apt-get install python3

install edgetpu for python3
  
         $ Sudo apt-get install python3-edgetpu
	 $ Sudo apt-get install edgetpu-examples

Test the installation of the Python API

	$ Python3 -c ‘import edgetpu; print(“ok”)’
	
This command should print "OK" and not show any error messages if everything worked.

If not, consult your device setup guide(https://coral.ai/docs/setup).

Install minigo

Start by downloading the Minigo Git repository:
	$ sudo apt update
	$ sudo apt install git
	$ git clone https://www.github.com/tensorflow/minigo
	
Then launch the included Python virtual environment and install other requirements:

	$ cd minigo
	$ pip3 install virtualenv virtualenvwrapper
	$ python3 -m virtualenv --system-site-packages ./
	$ source ./bin/activate
	$ sh ./minigui/edgetpu/install_requirements.sh

Continuing from inside of the minigo directory, download the Minigo model:

	$ mkdir saved_models
	$ curl http://storage.googleapis.com/minigo-pub/edgetpu-19x19/minigo-v17-2019-04-29-edgetpu.tflite -o saved_models/v17-2019-04-29-edgetpu.tflite
	
Start the Minigo server

Now launch the Minigo Server using the following command. (If you downloaded a model different from the one above, you need to edit the corresponding line in minigui/control/minigo_edgetpu.ctl.)

	$ python3 minigui/serve.py --control minigui/control/minigo_edgetpu.ctl &
The Minigo game server is now running, waiting for you to start a game. The ampersand (&) at the end of the above command makes the program run in the background, so you can still access the terminal prompt, which you'll want for the commands below.

Start the game:

If you just want to watch the AI play against itself,
start the kiosk mode by opening this URL in a browser on the host computer: http://localhost:5001/kiosk.html.
Note: We recommend use chrome or chromium for better result

If you're using the Coral Dev Board over SSH/MDT, connect a monitor to the HDMI port and then run this command from the Dev Board shell:

	$ sh ./minigui/edgetpu/start_chromium.sh http://localhost:5001/kiosk.html

You should then see the game appear on the monitor as shown in figure

![image](https://user-images.githubusercontent.com/53611350/140893709-03260032-f31e-4ac5-83b3-5cc1415c40d9.png)
Minigo playing in kiosk mode.

The large board on the left shows the current game pieces with opaque pieces and Minigo's "principal variation" with semi-transparent pieces. The principal variation is the sequence of moves Minigo currently considers to be the best for both players, and it may change for a period of time until Minigo selects its move.

To the right, there are three smaller boards (only in kiosk mode):

Top: The current move sequence that Minigo is considering. If it decides this is better than the sequence on the left, then the board on the left updates its principal variation.

Middle: A heat map of where Minigo is focusing its attention for the next move, indicated with dark squares. The darker a square is, the more time Minigo spends investigation variations that start with a move at that point.

Bottom: A heat map of "bad moves." That is, the darker an area is colored (in the color of the opponent), the more likely that a play in that area could swing the game in the favor of the opponent. (Except during the first handfull of moves, Minigo normally considers most points on the board to be bad moves, as demonstrated in the screenshot above: it's white's turn to play and Minigo thinks any move except the one at point O2 will swing the game in black's favor, which is why the board is tinted black everywhere except that point.)
	
	
Play a game versus Minigo

To play against the Minigo AI, instead navigate to http://localhost:5001/lw_demo.html.

If you're using the Coral Dev Board over MDT/SSH, connect a monitor to the HDMI port and connect a mouse to the USB-A port. Then run this command from the Dev Board shell:

	$ sh ./minigui/edgetpu/start_chromium.sh http://localhost:5001/lw_demo.html
	
By default, the game is set up for human vs. human gameplay: Notice at the top-right corner, there are two buttons that both say "Human" (see figure 2). These indicate which player (black and white) is controlled by a human.
	
![image](https://user-images.githubusercontent.com/53611350/140894532-81ae192e-7a3a-4553-8be0-5f96e18708d3.png)

The buttons to assign players and control gameplay

To play against Minigo, click one of the buttons to change the player to "Minigo." Remember that black plays first, so if you want to play first, click the white button to set white as "Minigo."

To play a piece, use your mouse to click a position on the board.

When it's Minigo's turn, you'll see the principal variation appear on the board (the semi-transparent pieces, as described above) while the AI model assesses the game board options. Once it decides, Minigo places a piece on the board and control returns to you so you can play.

Good Luck......

![image](https://user-images.githubusercontent.com/53611350/140894645-4c6b1c60-8daf-44c2-be3a-59e619352618.png)
Minigo playing in "lightweight demo" mode, which allows one or two human players


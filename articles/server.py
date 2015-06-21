__author__ = 'Stretchhog'

from flask import Flask, render_template
import cherrypy
from paste.translogger import TransLogger

app = Flask(__name__)
app.debug = True


@app.route("/")
def hello():
	return "Hello World!"


@app.route('/welcome')
def welcome():
	return render_template('welcome.html')


def run_server():
	# Enable WSGI access logging via Paste
	app_logged = TransLogger(app)

	# Mount the WSGI callable object (app) on the root directory
	cherrypy.tree.graft(app_logged, '/')

	# Set the configuration of the web server
	cherrypy.config.update({
		'engine.autoreload_on': True,
		'log.screen': True,
		'server.socket_port': 5050,
		'server.socket_host': '127.0.0.1'
	})

	# Start the CherryPy WSGI web server
	cherrypy.engine.start()
	cherrypy.engine.block()


if __name__ == "__main__":
	run_server()

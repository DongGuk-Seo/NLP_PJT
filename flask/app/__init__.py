from distutils.log import debug
from flask import Flask

def create_app():
    app = Flask(__name__)

    from .views import main_views
    app.register_blueprint(main_views.bp)

    if __name__ == '__main__':
        app.run(debug=True)
        
    return app

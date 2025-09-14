import sys
from PyQt5 import QtWidgets
from .folder_selector import FolderSelectorWidget


def run_app():
    app = QtWidgets.QApplication(sys.argv)
    win = FolderSelectorWidget()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()

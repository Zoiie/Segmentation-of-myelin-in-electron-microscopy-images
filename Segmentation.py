# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'openfoldertest.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
# from PyQt5.QtCore import pyqtSlot
import prediction_model
import sys

# path=""
class Ui_Form(object):
    def setupUi(self, Form):
        # @pyqtSlot(object)
        self.path = ""
        Form.setObjectName("Segmentation")
        Form.resize(470, 60)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(250, 10, 111, 31))
        self.pushButton.setObjectName("pushButton")

        self.pushButton1 = QtWidgets.QPushButton(Form)
        self.pushButton1.setGeometry(QtCore.QRect(130, 10, 111, 31))
        self.pushButton1.setObjectName("pushButton")

        # self.pushButton2 = QtWidgets.QPushButton(Form)
        # self.pushButton2.setGeometry(QtCore.QRect(250, 10, 111, 31))
        # self.pushButton2.setObjectName("pushButton")

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(370, 10, 111, 31))
        self.label.setObjectName("label")

        self.open_file_pushbutton = QtWidgets.QPushButton(Form)
        self.open_file_pushbutton.setGeometry(10, 10, 111, 31)
        self.open_file_pushbutton.setObjectName('open_pushbutton')


        self.retranslateUi(Form)
        # 点击按钮信号传送到打开文件夹函数

        self.open_file_pushbutton.clicked.connect(self.open)
        # path=self.open_file_pushbutton.
        self.pushButton.clicked.connect(self.openfolder)
        self.pushButton1.clicked.connect(self.segment)
        # self.pushButton2.clicked.connect(self.openfolder)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Segmentation", "EM Segmentation"))
        self.pushButton.setText(_translate("Form", "Results"))
        self.pushButton1.setText(_translate("Form", "Begin"))
        # self.pushButton2.setText(_translate("Form", "打开文件夹2"))
        self.open_file_pushbutton.setText(_translate("Form", 'Open Folder'))
        self.label.setText(_translate("Form", "By ZHOU Yu"))

    def open(self):
        file_path = QFileDialog.getExistingDirectory()
        ldr=os.listdir(file_path)
        # if file_path == None:
        print("ldr:",ldr)
        if ldr ==[]:
            self.messageDialog()
        else:
            self.initial_path = file_path  # self.initial_path用来存放图片所在的文件夹

        self.path = file_path
        print(self.path)

    def segment(self):
        # path=QFileDialog.getExistingDirectory()
        print("/////", self.path)
        prediction_model.main(self.path)

    def openfolder(self, Form):
        '''打开系统文件资源管理器的对应文件夹'''
        import os
        folder = self.path
        print("folder", folder)
        # # 方法1：通过start explorer
        # os.system("start explorer %s" % folder)
        # 方法2：通过startfile
        os.startfile(folder)

    def messageDialog(self):
        msg_box = QMessageBox(QMessageBox.Warning, 'Warning', 'No file in this folder! Please re-select!')
        msg_box.exec_()

# QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 适配2k等高分辨率屏幕，低分辨率屏幕可除去
app = QtWidgets.QApplication(sys.argv)
Form = QtWidgets.QWidget()
ui = Ui_Form()
ui.setupUi(Form)
Form.show()
sys.exit(app.exec_())
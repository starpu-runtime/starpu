# -------------------------------------------------
# Project created by QtCreator 2011-01-22T11:55:16
# -------------------------------------------------
QT += network
QT += opengl
QT += sql

TARGET = StarPU-Top
TEMPLATE = app
SOURCES += main.cpp \
#STARPU-TOP
    mainwindow.cpp \
    configurationmanager.cpp \
    widgetwindowsmanager.cpp \
    communicationthread.cpp \
    communicationmanager.cpp \
    preferencesdialog.cpp \
    datawidget.cpp \
    interactivewidget.cpp \
    ganttwidget.cpp \
    debugconsole.cpp \
    dataaggregatorwidget.cpp \
    taskmanager.cpp \
    abstractwidgetwindow.cpp \
    sessionsetupmanager.cpp \
#QLEDINDICATOR
    qledindicator/qledindicator.cpp \
    aboutdialog.cpp
HEADERS += mainwindow.h \
#STARPU-TOP
    starputoptypes.h \
    widgetwindowsmanager.h \
    configurationmanager.h \
    communicationthread.h \
    communicationmanager.h \
    preferencesdialog.h \
    datawidget.h \
    interactivewidget.h \
    ganttwidget.h \
    debugconsole.h \
    dataaggregatorwidget.h \
    taskmanager.h \
    abstractwidgetwindow.h \
    sessionsetupmanager.h \
#QLEDINDICATOR
    qledindicator/qledindicator.h \
    aboutdialog.h

FORMS += mainwindow.ui \
    preferencesdialog.ui \
    debugconsole.ui \
    aboutdialog.ui
RESOURCES += resources.qrc
OTHER_FILES += TODO.txt

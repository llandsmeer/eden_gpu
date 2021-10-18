//
// Created by max on 26-01-21.
// c++ LIB
// todo remove weird defines and make them const
// todo create files on the side
//

#include <utility>
#include <iostream>

#ifndef LOGGING_MINILOGGER_H
#define LOGGING_MINILOGGER_H

#define LOG_MAXFUNCLENGHT  12

#define LOG_OVERWRITE 0
#define LOG_ERR       1
#define LOG_WARN      2
#define LOG_MES       3
#define LOG_TIME      4
#define LOG_INFO      5
#define LOG_DEBUG     6
#define LOG_DEFAULT   5

#define ILOG_MES     "[MES   ]"
#define ILOG_TIME    "[TIME  ]"
#define ILOG_DEBUG   "[DEBUG ]"
#define ILOG_ERROR   "[ERROR ]"
#define ILOG_WARNING "[WARN  ]"
#define ILOG_INFO    "[INFO  ]"
#define ILOG_OVER    "        "

#define LOG_ENDL std::endl<char, std::char_traits<char>>

#define CLOG_OVER    "\033[0;0m"
#define CLOG_MES     "\033[0;37m"
#define CLOG_DEBUG   "\033[0;36m"
#define CLOG_TIME    "\033[0;35m"
#define CLOG_ERROR   "\033[0;31m"
#define CLOG_WARNING "\033[0;33m"
#define CLOG_INFO    "\033[0;34m"

#define INIT_LOG() miniLogger log(LOG_DEFAULT,std::cout, __FUNCTION__)
#define INIT_LOG_LL(a) miniLogger log(a,std::cout, __FUNCTION__)
#define INIT_LOGP(a,b) miniLogger log(a,std::cout, __FUNCTION__,b)

class miniLogger {
public:

    miniLogger():
            _enbMPI(false),
            _loglevel(LOG_WARN),
            _message_level(0),
            _processID(0),
            _fac(std::cout),
            _name("default")
    {};

    miniLogger(unsigned LL, std::ostream& f, std::string n):
            _enbMPI(false),
            _loglevel(LL),
            _message_level(0),
            _processID(0),
            _fac(f),
            _name(std::move(n))
    {};

    miniLogger(unsigned LL, std::ostream& f, std::string n, int& processID):
            _enbMPI(true),
            _loglevel(LL),
            _message_level(0),
            _processID(processID),
            _fac(f),
            _name(std::move(n))
    {};

    miniLogger(std::ostream& f, std::string n):
            _enbMPI(true),
            _loglevel(LOG_WARN),
            _message_level(0),
            _processID(0),
            _fac(f),
            _name(std::move(n))
    {};

    miniLogger(std::ostream& f, std::string n, int& processID):
            _enbMPI(true),
            _loglevel(LOG_WARN),
            _message_level(0),
            _processID(processID),
            _fac(f),
            _name(std::move(n))
    {};

    template<typename T>
    friend inline miniLogger& operator << (miniLogger& l, const T& s){
        if (l._message_level <= l._loglevel){
            l._fac << s;
            return l;
        }
        else{
            return l;
        }
    }
    inline miniLogger& operator()(unsigned ll){
        _message_level = ll;
        if (_message_level <= _loglevel) {
            if(_enbMPI)
                _fac << prep_color() << prep_level() << prep_name() << prep_process() << " : \033[0m";
            else
                _fac << prep_color() << prep_level() << prep_name() << " : \033[0m";
        }
        return *this;
    }

    [[maybe_unused]] inline void skipline(){
        _fac << LOG_ENDL;
    }

    bool set_processID(uint ID){
        _processID = ID;
        return true;
    }

    bool mpi(){
        _enbMPI = true;
        return true;
    }

    bool set_name(std::string& name){
        _name = name;
        return true;
    }
    inline miniLogger& operator()(unsigned ll, unsigned PID) {
        _message_level = ll;
        if (_message_level <= _loglevel && PID == _processID){
            if(_enbMPI)
                _fac << prep_color() << prep_level() << prep_name() << prep_process() << " : \033[0m";
            else
                _fac << prep_color() << prep_level() << prep_name() << " : \033[0m";
        } else {
            _message_level = 100;
        }
        return *this;
    }
    bool set_Loglevel(int ll){
        _loglevel = ll;
        return true;
    };
private:
    bool _enbMPI;
    unsigned _loglevel ;
    unsigned _message_level;
    uint _processID;
    std::ostream& _fac;
    std::string _name;
    std::string prep_name(){
        if (_message_level != LOG_OVERWRITE) {
            std::string temp_string = "[ " + _name;
            int tmp_ToAdd = LOG_MAXFUNCLENGHT - _name.length();
            if(tmp_ToAdd>=0){
                for(int i = 0; i < tmp_ToAdd; i++)
                    temp_string += " ";
            }
            else{
                temp_string.resize(LOG_MAXFUNCLENGHT);
                temp_string += "..";
            }
            temp_string += " ]";
            return temp_string;
        }
        std::string temp = "    ";
        for(int i = 0; i<LOG_MAXFUNCLENGHT;i++) {
            temp += " ";
        }
        return temp;
    }
    [[nodiscard]] std::string prep_color() const {
        switch (_message_level)
        {
            case LOG_MES:
                return CLOG_MES;
            case LOG_ERR:
                return CLOG_ERROR;
            case LOG_WARN:
                return CLOG_WARNING;
            case LOG_INFO:
                return CLOG_INFO;
            case LOG_DEBUG:
                return CLOG_DEBUG;
            case LOG_TIME:
                return CLOG_TIME;
            case LOG_OVERWRITE:
                return CLOG_OVER;
            default:
                return "";
        }
    }
    [[nodiscard]] std::string prep_level() const {
        switch (_message_level)
        {
            case LOG_OVERWRITE:
                return ILOG_OVER;
            case LOG_MES:
                return ILOG_MES;
            case LOG_ERR:
                return ILOG_ERROR;
            case LOG_WARN:
                return ILOG_WARNING;
            case LOG_INFO:
                return ILOG_INFO;
            case LOG_DEBUG:
                return ILOG_DEBUG;
            case LOG_TIME:
                return ILOG_TIME;
            default:
                return "";
        }
    }
    [[nodiscard]] std::string prep_process() const{
        std::string temp = "[ " + std::to_string(_processID) + " ]";
        return temp;
    }
};

#endif //LOGGING_MINILOGGER_H
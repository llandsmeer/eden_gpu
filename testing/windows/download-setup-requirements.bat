
call %~dp0\default-tooling-path.bat

set DOWNLOADS_DIR=%TOOLING_DIR%\downloads

set WGET_DIR=%TOOLING_DIR%\wget
set WGET=%WGET_DIR%\wget.exe

set ZIP7_DIR=%TOOLING_DIR%\7-Zip
set ZIP7=%ZIP7_DIR%\7z.exe

set COREUTILS_DIR=%TOOLING_DIR%\coreutils
set UNXUTILS_DIR=%TOOLING_DIR%\unxutils
set FLEXBISON_DIR=%TOOLING_DIR%\flexbison
set XXD_DIR=%TOOLING_DIR%\xxd
set MAKE_DIR=%TOOLING_DIR%\make
set MINGW32_DIR=%TOOLING_DIR%\mingw32
set MINGW64_DIR=%TOOLING_DIR%\mingw64
set PYTHON32_DIR=%TOOLING_DIR%\python32
set PYTHON64_DIR=%TOOLING_DIR%\python64

set GETPIP_DIR=%DOWNLOADS_DIR%


md %DOWN_REQS_DIR%

:: get wget 
:: tls 1.2 is not on by default: https://stackoverflow.com/questions/49800534/powershell-could-not-create-ssl-tsl-secure
:: Also note that PowerShell may not be configured to run files (even though it always runs one-liners) and piping them instead breaks error handling (powershell thinks it's interactive output)
set Download_Prefix=[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest

md %WGET_DIR%
powershell -command %Download_Prefix% https://eternallybored.org/misc/wget/1.21.1/32/wget.exe -ErrorAction Stop -OutFile "%WGET_DIR%/wget.exe" || goto :error

:: download the rest of the files
md %DOWNLOADS_DIR%
:: "%WGET_DIR%/wget.exe" --input-file=files-to-get.txt --directory-prefix=%DOWNLOADS_DIR% || goto :error
:: either of --backups=1 --no-clobber leads to redirect loop, oh well
%WGET% -O %DOWNLOADS_DIR%\7z1900.msi https://7-zip.org/a/7z1900.msi || goto :error
:: because wget fails to overwrite files in their names are supplied using a list...
%WGET% -O %DOWNLOADS_DIR%\coreutils-5.3.0-bin.zip https://sourceforge.net/projects/gnuwin32/files/coreutils/5.3.0/coreutils-5.3.0-bin.zip || goto :error
%WGET% -O %DOWNLOADS_DIR%\coreutils-5.3.0-dep.zip https://sourceforge.net/projects/gnuwin32/files/coreutils/5.3.0/coreutils-5.3.0-dep.zip || goto :error
%WGET% -O %DOWNLOADS_DIR%\unxutils.zip https://sourceforge.net/projects/unxutils/files/unxutils/current/UnxUtils.zip || goto :error

%WGET% -O %DOWNLOADS_DIR%\findutils-4.2.20-2-bin.zip https://sourceforge.net/projects/gnuwin32/files/findutils/4.2.20-2/findutils-4.2.20-2-bin.zip || goto :error
%WGET% -O %DOWNLOADS_DIR%\win_flex_bison-2.5.18.zip https://sourceforge.net/projects/winflexbison/files/win_flex_bison-2.5.18.zip || goto :error
%WGET% -O "%DOWNLOADS_DIR%\xxd-1.11_win32(static).zip" "https://sourceforge.net/projects/xxd-for-windows/files/xxd-1.11_win32%%28static%%29.zip" || goto :error

%WGET% --secure-protocol=TLSV1_3 -O %DOWNLOADS_DIR%\gnumake-4.3.exe https://github.com/mbuilov/gnumake-windows/blob/master/gnumake-4.3.exe?raw=true || goto :error

%WGET% -O %DOWNLOADS_DIR%\mingw32.7z "https://sourceforge.net/projects/mingw-w64/files/Toolchains targetting Win32/Personal Builds/mingw-builds/8.1.0/threads-posix/sjlj/i686-8.1.0-release-posix-sjlj-rt_v6-rev0.7z" || goto :error
%WGET% -O %DOWNLOADS_DIR%\mingw64.7z "https://sourceforge.net/projects/mingw-w64/files/Toolchains targetting Win64/Personal Builds/mingw-builds/8.1.0/threads-posix/seh/x86_64-8.1.0-release-posix-seh-rt_v6-rev0.7z" || goto :error

%WGET% -O %DOWNLOADS_DIR%\python-win32.7z https://www.python.org/ftp/python/3.5.2/python-3.5.2-embed-win32.zip || goto :error
%WGET% -O %DOWNLOADS_DIR%\python-amd64.7z https://www.python.org/ftp/python/3.5.2/python-3.5.2-embed-amd64.zip || goto :error


:unpack
:: unpack and customise the various files

:: 7-zip
md %ZIP7_DIR%
call :realpath %ZIP7_DIR%
msiexec /a %DOWNLOADS_DIR%\7z1900.msi /qn TARGETDIR=%retval% || goto :error
copy "%ZIP7_DIR%\Files\7-Zip\*" %ZIP7_DIR%

:: GNU Coreutils
%ZIP7% -y x %DOWNLOADS_DIR%\coreutils-5.3.0-bin.zip -o%COREUTILS_DIR% || goto :error
%ZIP7% -y x %DOWNLOADS_DIR%\coreutils-5.3.0-dep.zip -o%COREUTILS_DIR% || goto :error
:: add also findutils while at it
%ZIP7% -y x %DOWNLOADS_DIR%\findutils-4.2.20-2-bin.zip -o%COREUTILS_DIR% || goto :error

:: UnxUtils
%ZIP7% -y x %DOWNLOADS_DIR%\unxutils.zip -o%UNXUTILS_DIR% || goto :error
:: replace rm of Coreutils, because its globbing is broken on Win6+
:: https://stackoverflow.com/a/10683011
copy /Y %UNXUTILS_DIR%\usr\local\wbin\rm.exe %COREUTILS_DIR%\bin\rm.exe


:: flex, bison
%ZIP7% -y x %DOWNLOADS_DIR%\win_flex_bison-2.5.18.zip -o%FLEXBISON_DIR% || goto :error
move /Y %FLEXBISON_DIR%\win_bison.exe %FLEXBISON_DIR%\bison.exe || goto :error
move /Y %FLEXBISON_DIR%\win_flex.exe %FLEXBISON_DIR%\flex.exe || goto :error

::xxd
:: has top level folder, fortunately it's just one file
%ZIP7% -y e "%DOWNLOADS_DIR%\xxd-1.11_win32(static).zip" -o%XXD_DIR% "*/*" || goto :error

:: GNU Make
md %MAKE_DIR% 
:: rename to make.exe while at it
copy /Y %DOWNLOADS_DIR%\gnumake-4.3.exe %MAKE_DIR%\make.exe || goto :error

:: MinGW
:: use parent directory because these have a top level folder, don't just assume that folder will always have the same name as the zip file...
:: but assume the folders are mingw32 and mingw64, for now
%ZIP7% -y x %DOWNLOADS_DIR%\mingw32.7z -o%MINGW32_DIR%\.. || goto :error
%ZIP7% -y x %DOWNLOADS_DIR%\mingw64.7z -o%MINGW64_DIR%\.. || goto :error

:: Python
%ZIP7% -y x %DOWNLOADS_DIR%\python-win32.7z -o%PYTHON32_DIR% || goto :error
%ZIP7% -y x %DOWNLOADS_DIR%\python-amd64.7z -o%PYTHON64_DIR% || goto :error
:: also copy the binaries to "python3", it's not bad because this is an one-off setup
copy /Y %PYTHON32_DIR%\python.exe %PYTHON32_DIR%\python3.exe || goto :error
copy /Y %PYTHON64_DIR%\python.exe %PYTHON64_DIR%\python3.exe || goto :error

:: now setup both pythons, with pip and latest packages to build wheels
%WGET% -O %DOWNLOADS_DIR%\get-pip.py https://bootstrap.pypa.io/pip/3.5/get-pip.py || goto :error

call :setup_python %PYTHON32_DIR% %GETPIP_DIR%
call :setup_python %PYTHON64_DIR% %GETPIP_DIR%

@goto :done_python
:setup_python
:: 1 is location of python, 2 is location of get-pip.py
"%1\python3" "%2\get-pip.py" || goto :error
"%1\python3" -m pip install -U pip setuptools auditwheel twine || goto :error

@exit /b
:done_python


:: finished !
@exit /b 0


:: useful subroutines

::
:realpath
set retval=%~f1
@exit /b


:: on error
:error
@echo ERROR : Last command failed with error code %errorlevel%. Exiting.
@exit /b %errorlevel%


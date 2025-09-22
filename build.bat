@echo off
REM clean
if exist site rmdir /s /q site
mkdir site

REM build HPR-LP
sphinx-build -b html docs\hprlp site\hprlp

REM build HPR-QP
sphinx-build -b html docs\hprqp site\hprqp

REM copy landing
copy /y landing\index.html site\index.html

echo Done. Open site\index.html

pause

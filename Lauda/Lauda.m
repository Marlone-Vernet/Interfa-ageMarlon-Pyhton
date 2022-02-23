lauda = serial('COM9','BaudRate',9600);
fopen(lauda);
set(lauda,'Terminator','CR/LF')
fprintf(lauda,'START')
txt_temp = ['OUT_SP_00_',num2str(25.0)]; % In degree celsius

fprintf(lauda,txt_temp)
fprintf(lauda, 'OUT_SP_01_2')

pause
fprintf(lauda,'STOP')
fclose(lauda)


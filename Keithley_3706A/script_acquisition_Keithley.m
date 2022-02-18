clear

% First acquisition ? 

new=instrfind;
fclose(new);

% Acquisition parameters

dt = 10; %time between two points (without counting the switching time 
T = 86400; % Average duration of the measure

if dt == 0
    N = round(T/0.2); % number of iteration
else
    N = round(T/dt);
end


% Let us begin

obj1 = visa('ni','GPIB0::16::INSTR');  %adresse du keithley
fopen(obj1); 
%

fprintf(obj1,'reset()')


% fprintf(obj1, 'dmm.func("1032","temperature")')
% %%
% fprintf(obj1, 'dmm.transducer("1032",dmm.TEMP_THERMOCOUPLE)')
% %%
% fprintf(obj1, 'dmm.thermocouple("1032",dmm.THERMOCOUPLE_K)')
% %%
% fprintf(obj1, 'dmm.configure.set("1032","my_temp_k")')
% %%
% fprintf(obj1, 'dmm.setconfig("1032", "my_temp_k")')

% fprintf(obj1,'dmm.setconfig("1032,1048,1034,1050","twowireohms")')
% fprintf(obj1,'dmm.setconfig("1001,1018,1020","dcvolts")')
fprintf(obj1,'dmm.setconfig("1001","dcvolts")')


% fprintf(obj1, 'dmm.setconfig("1031",dmm.TEMPERATURE)')
% %%
% fprintf(obj1, 'dmm.transducer("1031",dmm.TEMP_THERMOCOUPLE)')
% %%
% fprintf(obj1, 'dmm.thermocouple("1031",dmm.THERMOCOUPLE_K)')

%%

% fprintf(obj1,'scan.create("1032,1048,1034,1050,1001,1018,1020,1002,1004")')
% fprintf(obj1,'scan.create("1031")')
fprintf(obj1,'scan.create("1001")')
% fprintf(obj1,'scan.create("1032,1048,1034,1050,1001,1018,1020")')

% fprintf(obj1,'scan.create("1002,1004")')


% fprintf(obj1, 'autodelay("dmm.OFF")')
% fprintf(obj1, ['channel.setdelay("1032", ' num2str(1/fs) ')'])
% fprintf(obj1, ['channel.setdelay("1048", ' num2str(1/fs) ')'])
% fprintf(obj1, ['channel.setdelay("1034", ' num2str(1/fs) ')'])
% fprintf(obj1, ['channel.setdelay("1050", ' num2str(1/fs) ')'])




a=0.001285;
b=0.0002362;
c=0.00000009285;


time = zeros(1, N);
tic 

for i= 1:N

    fprintf(obj1,'RBF = dmm.makebuffer(1000)')
    fprintf(obj1,'scan.execute(RBF)')
    fprintf(obj1,'printbuffer(1,RBF.n,RBF)')
    V(i,:)=str2num(fgets(obj1));

    time(1, i) = toc;
    
    pause(dt)
end

time = time-time(1);
 
%     for j=1:4
%         temperature(i,j)=1/(a+b*log(abs(V(i,j)))+c*log(abs(V(i,j)))^3)-273.15;
%     end

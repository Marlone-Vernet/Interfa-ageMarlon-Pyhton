clear
a=0.001285;
b=0.0002362;
d=0.00000009285;

c=clock;
 file=[num2str(c(3)),'-',num2str(c(2)),'-',num2str(c(1)),'_',num2str(c(4)),'h',num2str(c(5)),'.mat'];
new=instrfind;
fclose(new)
obj1=visa('ni','GPIB0::09::INSTR');
fopen(obj1)
obj2=visa('ni','GPIB0::05::INSTR');
fopen(obj2);
pause(0*3600)
fprintf(obj2,'VOLT 4.0')
 k=1;
 t(1)=0;
   pause(1*3600)
 fprintf(obj2,'VOLT 5')
 pause(1*3600)
  fprintf(obj2,'VOLT 6.5')
  pause(1*3600)
  
while l
%while t(k)<6000
    tic
   
        for kk=0:50
%         if t(end)<2*3600
%             fprintf(obj2,['VOLT ',num2str(5)])
        if t(end)>kk*3600*1 && t(end)<(kk+1)*3600*1 && kk<50
            fprintf(obj2,['VOLT ',num2str(7-kk*0.1)])
            %fprintf(obj2,['VOLT ',num2str(6)])
%         if t(end)>kk*3600*1 && t(end)<(kk+1)*3600*1 && kk>22 && kk<35
%              fprintf(obj2,['VOLT ',num2str(5.9+(kk-23)*0.1)])
%         elseif t(end)>kk*3600*1 && t(end)<(kk+1)*3600*1 && kk>24 
%              fprintf(obj2,['VOLT ',num2str(7.4-(kk-24)*0.1)])
        
        elseif t(end)>51*3600
            fprintf(obj2,'VOLT 0')
            break
        end
    end

   % fprintf(obj1,'*RST')
    fprintf(obj1,':SYST:BEEP:STAT OFF')
    %fprintf(obj1,'TRAC:CLE')
    %fprintf(obj1,'INIT:CONT OFF')
    %fprintf(obj1,'TRIG:SOUR IMM')
    %fprintf(obj1,'TRIG:COUN 1')
    %fprintf(obj1,'SAMP:COUN 4')
    fprintf(obj1,'FUNC "RES",(@101:105)')
    fprintf(obj1,'FUNC "VOLT",(@106)')
    fprintf(obj1,'ROUT:SCAN (@101:106)')
    %fprintf(obj1,'ROUT:SCAN:TSO IMM')
    %fprintf(obj1,'ROUT:SCAN:LSEL INT')
    fprintf(obj1,'READ?')
    A=fgets(obj1);
    B=split(A,',');
    

for i=1:1
r=str2num(B{1});
aa(k)=1/(a+b*log(r)+d*(log(r))^3)-273.15;
end

for i=1:1
r=str2num(B{2});
bb(k)=1/(a+b*log(r)+d*(log(r))^3)-273.15;
end

for i=1:1
r=str2num(B{3});
cc(k)=1/(a+b*log(r)+d*(log(r))^3)-273.15;
end

for i=1:1
r=str2num(B{4});
dd(k)=1/(a+b*log(r)+d*(log(r))^3)-273.15;
end

for i=1:1
r=str2num(B{5});
ee(k)=1/(a+b*log(r)+d*(log(r))^3)-273.15;
end

for i=1:1
r=str2num(B{6});
ff(k)=r;
end

fprintf(obj2,'MEASure:VOLT?')
tension(k)=str2num(fgets(obj2));
fprintf(obj2,'MEASure:CURRENT?')
courant(k)=str2num(fgets(obj2));
puissance(k)=courant(k)*tension(k);

figure(222);subplot(2,2,1);
plot(t,aa)
subplot(2,2,2);plot(t,bb)
subplot(2,2,3);plot(t,cc);
subplot(2,2,4);plot(t,dd);
% figure(88);plot(t/3600,ee)
 figure(89);plot(t/3600,ff)
%  if k>1000
%      figure(92);plot(t(end-1000:end)/3600,ff(end-1000:end))
%  end

% figure(223);clf;plot(t,c)
% figure(224);clf;plot(t,d)
 pause(0.1)
% if mod(k,100)==0
%     save(nom_fichier)
% end
k=k+1;
save(file)
tt=toc;
t(k)=t(k-1)+tt;
end

% mean(aa)
% mean(bb)
% std(aa)
% std(bb)   
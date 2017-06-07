%% Material Data from Book
       %1        2       3        4        5      6        7
       %Eglass   Boron   CarbonHT CarbonIM Kevlar Caqrbon  GraphiteHM
Data0= [0.127,    0.220,  0.129,    0.129,  0.132,   0.127,  0.127;  %1  %Thickness (mm)
	       40,      210,    136,      151,     75,     147,    181;  %2  %E1(GPa)Longitudinal Elestic Modulus
	      9.8,       20,     10,      9.4,      6,       9,   10.3;  %3  %E2(GPa)Transverse elastic modulus
	      2.8,        6,    5.2,      4.8,      2,     3.3,   7.17;  %4  %G12(GPa)Shear modulus
	     0.30,     0.30,   0.30,     0.31,   0.34,    0.31,   0.28;  %5  %nu12 Major Poisson’s ratio
	     1100,     1400,   1800,     2260,   1400,    2260,   1500;  %6  %sig1T(MPa)Ultimate longitudinal tensile strength
	      600,     2800,   1200,     1200,    280,    1200,   1500;  %7  %sig1C(MPa)Ultimate longitudinal compressive strength
	       20,       80,     40,       50,     30,      50,     40;  %8  %sig2t(MPa)Ultimate transverse tensile strength
	      140,      280,    220,      190,    140,     190,    246;  %9  %sig2c(MPa)Ultimate transverse compressive strength
           70,      120,     80,      100,     60,     100,     68;  %10 %tau12(MPa)Ultimate in-plane shear strength
        0.028,    0.007,  0.013,    0.015,  0.019,   0.015,   0.0083;%11 %eps1tbar       
	    0.015,    0.013,  0.009,    0.008,  0.004,   0.008,   0.0083;%12 %eps1cbar
	    0.002,    0.004,  0.009,    0.005,  0.005,   0.005,   0.0039;%13 %eps2tbar
	    0.014,    0.014,  0.004,    0.020,  0.023,   0.021,   0.0239;%14 %eps2cbar
        0.014,    0.020,  0.015,    0.022,  0.030,   0.030,   0.0095;%15 %tau12bar
        0.028,    0.007,  0.013,    0.015,  0.019,   0.015,   0.02;  %16 %alpha2(mikro m/m/°C)Longitudinal coefficient ofthermal expansion        
	    0.028,    0.007,  0.013,    0.015,  0.019,   0.015,   22.5;  %17 %alpha2(mikro m/m/°C)Transverse coefficient of thermal expansion
	    0.015,    0.013,  0.009,    0.008,  0.004,   0.008,   0.00;  %18 %beta2(Longitudinal coefficient of moisture expansion)
	    0.002,    0.004,  0.004,    0.005,  0.005,   0.005,   0.6;   %19 %beta2(Transverse coefficient of moisture expansion)
         1940,     1860,   1470,     1610,   1300,    1500,   1600];  %20 %density (kg/m^3)                

%Material Data from Exam
       %1        2       3        4        5      6        7
       %Eglass   Boron   CarbonHT CarbonIM Kevlar Caqrbon  GraphiteHM
Data1= [  0.5,    0.220,  0.129,    0.129,  0.132,   0.5,    0.5; %1  %Thickness (1)
	       40,      210,    136,      151,     75,     147,    181; %2  %E1(GPa)Longitudinal Elestic Modulus
	       10,       20,     10,      9.4,      6,       9,   10.3; %3  %E2(GPa)Transverse elastic modulus
	      2.8,        6,    5.2,      4.8,      2,     3.3,   7.17; %4  %G12(GPa)Shear modulus
	     0.25,     0.30,   0.30,     0.31,   0.34,    0.31,   0.28; %5  %nu12 Major Poisson’s ratio
	     1100,     1400,   1800,     2260,   1400,    2260,   1500; %6  %sigma1T(MPa)Ultimate longitudinal tensile strength
	      600,     2800,   1200,     1200,    280,    1200,   1500; %7  %sigma1C(MPa)Ultimate longitudinal compressive strength
	       20,       80,     40,       50,     30,      50,     40; %8  %sigma2t(MPa)Ultimate transverse tensile strength
	      140,      280,    220,      190,    140,     190,    246; %9  %sigma2c(MPa)Ultimate transverse compressive strength
           70,      120,     80,      100,     60,     100,     68; %10 %tau12(MPa)Ultimate in-plane shear strength
        0.028,    0.007,  0.013,    0.015,  0.019,   0.015,   0.0083;%11 %eps1tbar       
	    0.015,    0.013,  0.009,    0.008,  0.004,   0.008,   0.0083;%12 %eps1cbar
	    0.002,    0.004,  0.009,    0.005,  0.005,   0.005,   0.0039;%13 %eps2tbar
	    0.014,    0.014,  0.004,    0.020,  0.023,   0.021,   0.0239;%14 %eps2cbar
        0.014,    0.020,  0.015,    0.022,  0.030,   0.030,   0.0095;%15 %tau12bar
        0.028,    0.007,  0.013,    0.015,  0.019,   0.015,   0.02;  %16 %alpha2(mikro m/m/°C)Longitudinal coefficient ofthermal expansion        
	    0.028,    0.007,  0.013,    0.015,  0.019,   0.015,   22.5;  %17 %alpha2(mikro m/m/°C)Transverse coefficient of thermal expansion
	    0.015,    0.013,  0.009,    0.008,  0.004,   0.008,   0.00;  %18 %beta2(Longitudinal coefficient of moisture expansion)
	    0.002,    0.004,  0.004,    0.005,  0.005,   0.005,   0.6;   %19 %beta2(Transverse coefficient of moisture expansion)
         1940,     1860,   1470,     1610,   1300,    1500,   1600];  %20 %density (kg/m^3) 
%% Parameters
L=4000; %mm
b=1300; %m
e=400;  %m
q=0.005; %N/mm^2
m=3750; %kg
g=9.81;%m/s^2
teta=30;

Nmaterial= size(Data1,2);

% Nx=m*g*cosd(teta)/b;
Nx=400; %N/mm
Ny=200; %N/mm
% Ny=0; %N/mm
Nxy=0;

r=1;
%% Change Orientation and material
% for a=-90:1:90
Data2=[45,45
       1,1];
Data3=[45,45
       1,1];
% Data2=[0,90,45,-45,-45,45,90,0;
%        1,1,1,1,1,1,1,1];
% Data3=[0,90,45,-45,-45,45,90,0;
%        1,1,1,1,1,1,1,1];

Nlam=cumsum(Data2(2,:));
Nlamina=Nlam(1,end);

Nlayer=size(Data2,2);
Ncase=size(Data2,1);

numb=1;
Number=1;
%% Pick column and row based on material in the data, in this case is Epoxy
ColumnMat=6;
RowMat=1;

%Variable data
thick=Data1(RowMat,ColumnMat);
E1=1e3*Data1(RowMat+1,ColumnMat);
E2=1e3*Data1(RowMat+2,ColumnMat);
G12=1e3*Data1(RowMat+3,ColumnMat);
nu12=Data1(RowMat+4,ColumnMat);
nu21=(nu12/E1)*E2;
sigma1T=Data1(RowMat+5,ColumnMat);
sigma1C=Data1(RowMat+6,ColumnMat);
sigma2T=Data1(RowMat+7,ColumnMat);
sigma2C=Data1(RowMat+8,ColumnMat);

tau12=Data1(RowMat+9,ColumnMat);
eps1T=Data1(RowMat+10,ColumnMat);
eps1C=Data1(RowMat+11,ColumnMat);
eps2T=Data1(RowMat+12,ColumnMat);
eps2C=Data1(RowMat+13,ColumnMat);
gamma12=Data1(RowMat+14,ColumnMat);
alpha1=Data1(RowMat+15,ColumnMat);
alpha2=Data1(RowMat+16,ColumnMat); 
beta1=Data1(RowMat+17,ColumnMat);
beta2=Data1(RowMat+18,ColumnMat);

%Q local
Ql11 = E1/(1-nu12*nu21)     ;
Ql12 = nu21*E1/(1-nu12*nu21);
Ql21 = nu12*E2/(1-nu12*nu21);
Ql22 = E2/(1-nu12*nu21)     ;
Ql66 = G12                  ;

Ql=[Ql11,Ql12,0
        Ql21,Ql22,0
        0,0,Ql66]; %in Pa-m
%S local
Sl11 = 1/E1   ;
Sl12 = -nu21/E2;
Sl21 = -nu12/E1;
Sl22 = 1/E2   ;
Sl66 = 1/G12    ;

Sl=[Sl11,Sl12,0
    Sl12,Sl22,0
    0,0,Sl66]; %in Pa-m
sum=0;
t=0;
B=0;
D=0;
t1=1;
t2=0;
A=0;
n=1;
Ex=0;
Gxy=0;
%% Iteration for orientation
for p=1:Nlayer
while Data2(2,p)>0;
Data2(2,p)=Data2(2,p)-1;
for k=[Data2(1,p)];
disp(['orientation:',num2str(k),' degree'])
T=[cosd(k)^2        ,sind(k)^2       ,      -2*sind(k)*cosd(k); %Sigx=T*Sig1
   sind(k)^2        ,cosd(k)^2       ,       2*sind(k)*cosd(k); %Sigy=T*Sig2
   sind(k)*cosd(k)  ,-sind(k)*cosd(k),    cosd(k)^2-sind(k)^2]; %Tauxy=T*tau12

Q_bar=T*Ql*T'; %in MPa

Q_bar11=(Ql11*cosd(k)^4)+(2*(Ql12+2*Ql66))*cosd(k)^2*sind(k)^2+(Ql22*sind(k)^4)

S=(inv(T))'*Sl*inv(T); %we can use this for short version

ti=-Nlamina*thick/2+t1*thick;
t1=t1+1;
tj=-Nlamina*thick/2+t2*thick;
t2=t2+1;
Anew=(Q_bar*((ti)-(tj)));
A=A+Anew; %in N/mm

B=B+1/2*(Q_bar*((ti)^2-(tj)^2)); % in N
Dnew=1/3*(Q_bar*((ti)^3-(tj)^3))
D=D+Dnew %in Nmm
Exnew=1/S(1,1);
Ex=Ex+Exnew;
Gxynew=1/S(3,3);
Gxy=Gxy+Gxynew;
end
end
end

t1=1;
t2=0;
for h=1:Nlayer
disp(Data3)

while Data3(2,h)>0;
Data3(2,h)=Data3(2,h)-1;
for c=[Data3(1,h)];
disp(['orientation:',num2str(c),' degree'])
T=[cosd(c)^2        ,sind(c)^2       ,      -2*sind(c)*cosd(c); %Sigx=T*Sig1
   sind(c)^2        ,cosd(c)^2       ,       2*sind(c)*cosd(c); %Sigy=T*Sig2
   sind(c)*cosd(k)  ,-sind(c)*cosd(c), cosd(c)^2-sind(c)^2]; %Tauxy=T*tau12
ti=(-Nlamina*thick)/2+t1*thick;
t1=t1+1;
tj=(-Nlamina*thick)/2+t2*thick;
t2=t2+1;

% Global  and Local Strain
eps=inv(A)*[Nx;Ny;Nxy];
epsl=T'*eps;
% Average (homogenised) elastic Ex and Gxy
nusxnew=S(3,1)/S(3,3);
% nuxy=-S21/S11
% nuxs=S61/S11
% Local and Global Stress
sigmal=Ql*epsl;
sigma=T*sigmal;

% Maximum stress criterion
% Sigma local 1
if sigmal(1,1)>0;
sigma1bar=sigma1T;
if sigmal(1,1)<sigma1bar
disp(['Max stress: sigma local 1= ',num2str(sigmal(1,1)),' MPa (',num2str(sigma1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max stress: sigma local 1= ',num2str(sigmal(1,1)),' MPa (',num2str(sigma1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end
if sigmal(1,1)<0;
sigma1bar=-sigma1C;
if sigmal(1,1)>sigma1bar
disp(['Max stress: sigma local 1= ',num2str(sigmal(1,1)),' MPa (',num2str(sigma1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max stress: sigma local 1= ',num2str(sigmal(1,1)),' MPa (',num2str(sigma1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end

% Sigma local 2
if sigmal(2,1)>0;
sigma2bar=sigma2T;
if sigmal(2,1)<sigma2bar
disp(['Max stress: sigma local 2= ',num2str(sigmal(2,1)),' MPa (',num2str(sigma2bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max stress: sigma local 2= ',num2str(sigmal(2,1)),' MPa (',num2str(sigma2bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end

if sigmal(2,1)<0;
sigma2bar=-sigma1C;
if sigmal(2,1)>sigma2bar
disp(['Max stress: sigma local 1= ',num2str(sigmal(2,1)),' MPa (',num2str(sigma2bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max stress: sigma local 1= ',num2str(sigmal(2,1)),' MPa (',num2str(sigma2bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end


% tau local 12
if abs(sigmal(3,1))<tau12
disp(['Max stress: tau local 12= ',num2str(sigmal(3,1)),' MPa (',num2str(tau12),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max stress: tau local 12= ',num2str(sigmal(3,1)),' MPa (',num2str(tau12),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end

% Maximum strain criterion
% epsilon local 1
if epsl(1,1)>0;
eps1bar=eps1T;
if epsl(1,1)<eps1bar
disp(['Max strain: eps local 1= ',num2str(epsl(1,1)),' (',num2str(eps1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max strain: eps local 1= ',num2str(epsl(1,1)),' (',num2str(eps1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end

if epsl(1,1)<0;
eps1bar=-eps1C;
if epsl(1,1)>eps1bar
disp(['Max strain: eps local 1= ',num2str(epsl(1,1)),' (',num2str(eps1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max strain: eps local 1= ',num2str(epsl(1,1)),' (',num2str(eps1bar),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end


% epsilon local 2
if epsl(2,1)>0;
eps2bar=eps1T;
if epsl(2,1)<eps2bar
disp(['Max strain: eps local 2= ',num2str(epsl(2,1)),' (',num2str(eps2bar),') in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max strain: eps local 2= ',num2str(epsl(2,1)),' (',num2str(eps2bar),') in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end

if epsl(2,1)<0;
eps2bar=-eps1C;
if epsl(2,1)>eps2bar
disp(['Max strain: eps local 2= ',num2str(epsl(2,1)),' (',num2str(eps2bar),') in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max strain: eps local 2= ',num2str(epsl(2,1)),' (',num2str(eps2bar),') in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end
end


% gamma 12
if abs(epsl(3,1))<gamma12
disp(['Max strain: gamma 12= ',num2str(epsl(3,1)),' (',num2str(gamma12),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is safe'])
else
disp(['Max strain: gamma 12= ',num2str(epsl(3,1)),' (',num2str(gamma12),')in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti),' is fail'])
end

% Tsai hill
tsai=(sigmal(1,1)^2/(sigma1T)^2)-(sigmal(1,1)*sigmal(2,1)/sigma1T^2)+(sigmal(2,1)^2/(sigma1C)^2)+(sigmal(3,1)^2/tau12^2);
if tsai<=1
disp(['Tsai Hill= ',num2str(tsai),' in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti), ' is safe'])
else
disp(['Tsai Hill= ',num2str(tsai),' in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti), ' is fail'])
end
disp(['SF= ',num2str(1/tsai),' in orientation ' ,num2str(c),' deg between ',num2str(tj),' and ',num2str(ti)])

end
end
end

ExAvg=Ex/Nlamina; 
disp(['ExAvg= ',num2str(ExAvg),' MPa'])
GxyAvg=Gxy/Nlamina;
disp(['GxyAvg= ',num2str(GxyAvg),' MPa'])
combine=[A B;B D];

% To get zero displacement
for rho=Data1(20,1:Nmaterial) 
%disp(['Material for density:',num2str(rho),' '])
vblade=775.6/rho;
thick_blade=775.6*1e3/(rho*L*b);
Nply=thick_blade/0.127; %how much ply for material 1
end

% % Deflection and Twist
% x=[0:10:L];
% Mx=((-m*g*sind(teta)/b)*(L-x))+q/2*(L-x).^2
% int2Mx=cumtrapz(cumtrapz(x,Mx));
% Nint2Mx=size(int2Mx,2);
% 
% y=[0:10:L];
% My=0;
% int2My=My*y;
% Nint2My=size(int2My,2);
% 
% xy=[0:10:L];
% Mxy=-e*m*g*sind(teta)/(2*b);
% int2Mxy=Mxy*xy.^2/2;
% Nint2Mxy=size(int2Mxy,2);
% 
% kappa=inv(D);

% n=[int2Mx(1,1:Nint2Mx)];
% m=[int2My(1,1:Nint2My)];
% o=[int2Mxy(1,1:Nint2Mxy)];
% 
% % wx=(kappa(1,1)*n)+(kappa(1,2)*m)+(kappa(1,3)*o); %in mm
% % wy=(kappa(2,1)*n)+(kappa(2,2)*m)+(kappa(2,3)*o); %in mm
% % wxy=(kappa(3,1)*n)+(kappa(3,2)*m)+(kappa(3,3)*o);%in mm
% 
% x=[0:10:L];
% W =   -(((x.^4*kappa(1,1)*(q/2)/12))+ x.^3*kappa(1,1).*(m*g*sind(teta)/b) - (q*L)/6 + x.^2.*(kappa(1,1).*(L.^2*q/2 - m*g*sind(teta).*L/b)) + kappa(1,3)*(-e*m*g*sind(teta)/(2*b))/2);
% Tw = -0.5.*(x.^3*kappa(3,1).*(q/2)/3 + x.^2.*kappa(3,1).*(m*g*sind(teta)/b - q*L)/2 + x.*(kappa(3,1).*(L.^2.*q/2 - m*g*sind(teta)*L/b) + kappa(3,3).*(-e*m*g.*sind(teta)/(2*b))))*180/pi ;

% plot(x,wx)
% xlabel('x axis');
% ylabel('displacement x in mm');
% title('Displacement x vs. length');
% grid on
% 
% % figure()
% plot(y,wy)
% xlabel('y axis');
% ylabel('displacement y in mm');
% title('Displacement y vs length');
% grid on
% 
% % figure()
% plot(xy,wxy)
% xlabel('xy axis');
% ylabel('twist degree');
% title('twist vs. length');
% grid on
% 
% w=[wx;wy;wxy]
% 
% disp(['displacement   = ',num2str(W(1,end)),' mm'])
% disp(['twist          = ',num2str(Tw(1,end)),' deg'])

% syms x y real
d = inv(D);

c1 = q/2;
c2 = m*g*sind(teta)/b - q*L;
c3 = L^2*q/2 - m*g*sind(teta)*L/b; 
c4 = -e*m*g*sind(teta)/(2*b); %Mxy

x=[0:1:L];
def =  -(x.^4*d(1,1)*c1/12 + x.^3*d(1,1)*c2/6 + x.^2*(d(1,1)*c3 + d(1,3)*c4)/2) ;
Tw = -0.5*(x.^3*d(3,1)*c1/3 + x.^2*d(3,1)*c2/2 + x*(d(3,1)*c3 + d(3,3)*c4))*180/pi  ;


disp(['displacement   = ',num2str(max(def)),' mm'])
disp(['twist          = ',num2str(max(Tw)),' deg'])
disp(['total thickness= ',num2str(thick*Nlamina),' mm'])
MaxTw(1,r)=max(Tw);
MaxDef(1,r)=max(def(1,end));
if r<180;
r=r+1;
end

% orientation=-90:1:89;
% plot(orientation,MaxDef);
% xlabel('Orientation');
% ylabel('Maximum w in mm');
% title('Maximum displacement vs orientation');
% grid on
% hold on
% 
% indexmin = find(min(MaxDef) == MaxDef); 
% xmin = orientation(indexmin); 
% ymin = MaxDef(indexmin);
% 
% indexmax = find(max(MaxDef) == MaxDef);
% xmax = orientation(indexmax);
% ymax = MaxDef(indexmax);
% 
% strmin = ['Minimum = ',num2str(ymin)];
% text(xmin,ymin,strmin,'HorizontalAlignment','left');
% 
% strmax = ['Maximum = ',num2str(ymax)];
% text(xmax,ymax,strmax,'HorizontalAlignment','right');
% 
% 
% figure()
% plot(orientation,MaxTw)
% xlabel('Orientation');
% ylabel('Maximum Twist in degree');
% title('Maximum Twist vs orientation');
% grid on


% 
% % figure()
% plot(x,W)
% xlabel('x axis');
% ylabel('displacement y in mm');
% title('Displacement y vs length');
% grid on
% hold on


% % 
% figure()
% plot(x,Tw)
% xlabel('xy axis');
% ylabel('twist degree');
% title('twist vs. length');
% grid on
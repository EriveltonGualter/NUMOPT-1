

[iksz, ipszilon] = meshgrid(1:0.5:10,1:20);
ze = sin(iksz) + cos(ipszilon);
surf(iksz, ipszilon, ze)
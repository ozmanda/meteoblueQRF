# meteoblueQRF
QRF model for the Meteoblue Stadtklima project

Includes a class which performs dropset error calculation, in which each measurement station is left out of training once and instead used for testing,
meaning that the extra- and interpolation errors can be analysed. Feature distance can be measured for each station and compared to the error, allowing 
for quantisation of which feature distance correlates most to large prediction errors. 

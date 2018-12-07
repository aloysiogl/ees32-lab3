ei_n0 = read('ei_n0.txt');
normal = read('normal_ps.txt');
hamming = read('hamming_ps.txt');
cyclic_6x10 = read('cyclicCíclico {6, 10}_ps.txt');
cyclic_7x12 = read('cyclicCíclico {7, 12}_ps.txt');
cyclic_8x14 = read('cyclicCíclico {8, 14}_ps.txt');
cyclic_9x15 = read('cyclicCíclico {9, 15}_ps.txt');
cyclic_9x16 = read('cyclicCíclico {9, 16}_ps.txt');
matrix_improved_14x6 = read('matrix_improved_14x6');
matrix_improved_21x9 = read('matrix_improved_21x9');
matrix_improved_28x12 = read('matrix_improved_28x12');
matrix_improved_35x15 = read('matrix_improved_35x15');
conv_ham0 = read('convolutional_hamming_pol_0_ps.txt');
conv_ham1 = read('convolutional_hamming_pol_1_ps.txt');
conv_ham2 = read('convolutional_hamming_pol_2_ps.txt');
conv_exa0 = read('convolutional_exact_pol_0_ps.txt');
conv_exa1 = read('convolutional_exact_pol_1_ps.txt');
conv_exa2 = read('convolutional_exact_pol_2_ps.txt');
conv_euc0 = read('convolutional_euclidean_pol_0_ps.txt');
conv_euc1 = read('convolutional_euclidean_pol_1_ps.txt');
conv_euc2 = read('convolutional_euclidean_pol_2_ps.txt');
    
hold on
grid on
x = [-5:1:10];
%plot(x, pchip(ei_n0, normal, x))
%plot(x, pchip(ei_n0, hamming, x))
plot(x, polyval(polyfit(ei_n0(1:14), normal(1:14),7), x))
%plot(x, polyval(polyfit(ei_n0(1:14), hamming(1:14),3), x))
%plot(x, pchip(ei_n0, cyclic_6x10, x))
%plot(x, pchip(ei_n0, cyclic_7x12, x))
%plot(x, polyval(polyfit(ei_n0(1:13), cyclic_7x12(1:13),4), x))
%plot(x, pchip(ei_n0, cyclic_8x14, x))
%plot(x, polyval(polyfit(ei_n0(1:13), cyclic_8x14(1:13),4), x))
%plot(x, pchip(ei_n0, cyclic_9x15, x))
%plot(x, polyval(polyfit(ei_n0(1:12), cyclic_9x15(1:12),5), x))
%plot(x, pchip(ei_n0, cyclic_9x16, x))
%plot(x, pchip(ei_n0, matrix_improved_14x6, x))
%plot(x, polyval(polyfit(ei_n0(1:14), matrix_improved_14x6(1:14),4), x))
%plot(x, pchip(ei_n0, matrix_improved_21x9, x))
%plot(x, pchip(ei_n0, matrix_improved_28x12, x))
%plot(x, pchip(ei_n0, matrix_improved_35x15, x))
%plot(x, pchip(ei_n0, conv_ham0, x))
plot(x, polyval(polyfit(ei_n0(1:9), conv_ham0(1:9),7), x))
%plot(x, pchip(ei_n0, conv_ham1, x))
plot(x, polyval(polyfit(ei_n0(1:8), conv_ham1(1:8),5), x))
%plot(x, pchip(ei_n0, conv_ham2, x))
plot(x, polyval(polyfit(ei_n0(1:7), conv_ham2(1:7),4), x))
%plot(x, pchip(ei_n0, conv_exa0, x))
%plot(x, pchip(ei_n0, conv_exa1, x))
%plot(x, pchip(ei_n0, conv_exa2, x))
%plot(x, pchip(ei_n0, conv_euc0, x))
%plot(x, pchip(ei_n0, conv_euc1, x))
%plot(x, pchip(ei_n0, conv_euc2, x))
%plot(x, polyval(polyfit(ei_n0(1:5), conv_euc2(1:5),4), x))
xlim([-2 10])
ylim([-4 0])
title('Comparação códigos convolucionais')
ylabel('log(probabilidade de erro)')
xlabel('E_i/N_0 (dB)')
%legend('Não codificado', 'Comparação Hamming polinômio 1', 'Comparação Hamming polinômio 2', ...
%'Comparação Hamming polinômio 3', 'Comparação Exata polinômio 1', 'Comparação Exata polinômio 2', ...
%'Comparação Exata polinômio 3', 'Comparação Euclidiana polinômio 1', 'Comparação Euclidiana polinômio 2', ...
%'Comparação Euclidiana polinômio 3');
legend('Não codificado', 'Comparação Hamming polinômio 1', 'Comparação Hamming polinômio 2', ...
'Comparação Hamming polinômio 3');
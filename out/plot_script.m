ei_n0 = read('ei_n0.txt');
normal = read('normal_ps.txt');
hamming = read('hamming_ps.txt');
cyclic_8x14 = read('cyclicCíclico {8, 14}_ps.txt');
matrix_improved_14x6 = read('matrix_improved_14x6');
matrix_improved_21x9 = read('matrix_improved_21x9');
matrix_improved_28x12 = read('matrix_improved_28x12');
matrix_improved_35x15 = read('matrix_improved_35x15');
conv_ham0 = read('convolutional_hamming_pol_0_ps.txt');
conv_ham1 = read('convolutional_hamming_pol_1_ps.txt');
conv_ham2 = read('convolutional_hamming_pol_2_ps.txt');

hold on
x = [-5:1:10];
plot(x, pchip(ei_n0, normal, x))
plot(x, pchip(ei_n0, hamming, x))
plot(x, pchip(ei_n0, cyclic_8x14, x))
plot(x, pchip(ei_n0, matrix_improved_14x6, x))
plot(x, pchip(ei_n0, matrix_improved_21x9, x))
plot(x, pchip(ei_n0, matrix_improved_28x12, x))
plot(x, pchip(ei_n0, matrix_improved_35x15, x))
plot(x, pchip(ei_n0, conv_ham0, x))
plot(x, pchip(ei_n0, conv_ham1, x))
plot(x, pchip(ei_n0, conv_ham2, x))
legend('Não codificado', 'Hamming', 'improved\_14x6', ...
'improved\_21x9', 'improved\_28x12', 'improved\_35x15')
clc
clear
data = readtable('../data/train.csv', 'ReadVariableNames', true);

%% Asignar variables
y = data{:,1};              % Variable dependiente (0,1)
X1 = data{:,2};             % Valores numéricos (unidades)
X2 = data{:,3};             % Valores numéricos (décimas)
X3 = data{:,4};             % Días de la semana (categóricos)
X4 = data{:,5};             % Valores numéricos (unidades)
X5 = data{:,6};             % Valores numéricos (centenas)
X6 = data{:,7};             % Estados de EE.UU (categóricos)
X7 = data{:,8};             % Marcas de carros (categóricos)% Identificar y mostrar valores faltantes

summary(data);


missingData = ismissing(data);
figure;

% Crear una tabla para el heatmap
missingTable = array2table(missingData, 'VariableNames', data.Properties.VariableNames);

numMissing = sum(missingData);
% Calcular el porcentaje de datos faltantes por columna
numRows = size(data, 1);  % Número total de filas
percentMissing = (numMissing / numRows) * 100;

% Mostrar el porcentaje de datos faltantes por columna
disp(percentMissing)

%heatmap(missingTable, 'Mapa de Datos Faltantes', 'XLabel', 'Variables', 'YLabel', 'Observaciones');

% Calcular estadísticas
stats = varfun(@mean, data(:,[2 3 5 6]), 'OutputFormat', 'table'); % Media
disp(stats);

%% Visualizar distribución
%figure;
subplot(2,2,1); histogram(X1); title('Distribución de X1');
subplot(2,2,2); histogram(X2); title('Distribución de X2');
subplot(2,2,3); histogram(X4); title('Distribución de X4');
subplot(2,2,4); histogram(X5); title('Distribución de X5');

figure;
subplot(2,2,1); boxplot(X1); title('Boxplot de X1');
subplot(2,2,2); boxplot(X2); title('Boxplot de X2');
subplot(2,2,3); boxplot(X4); title('Boxplot de X4');
subplot(2,2,4); boxplot(X5); title('Boxplot de X5');

%% Distribución de X3 (días de la semana)
X3_categorical = categorical(X3); % Asegúrate de que X3 es categórico
dayCounts = countcats(X3_categorical); % Contar las ocurrencias

% Obtener los nombres únicos de las categorías (días de la semana)
dayNames = categories(X3_categorical);

% Graficar las frecuencias con bar
figure;
bar(dayCounts);
set(gca, 'XTickLabel', dayNames); % Asignar etiquetas a los ejes X
title('Distribución de Días de la Semana');
xlabel('Días de la Semana');
ylabel('Frecuencia');

%%  Distribución de X6 (estados de EE.UU.)
X6_categorical = categorical(X6); % Asegúrate de que X3 es categórico
stateCounts = countcats(X6_categorical); % Contar las ocurrencias

% Obtener los nombres únicos de las categorías (días de la semana)
StateNames = categories(X6_categorical);

% Graficar las frecuencias con bar
figure;
bar(stateCounts);
set(gca, 'XTickLabel', StateNames); % Asignar etiquetas a los ejes X
title('Distribución de estados');
xlabel('Estados');
ylabel('Frecuencia');

%% Distribución de X7 (marcas de carros)
X7_categorical = categorical(X7); % Asegúrate de que X3 es categórico
carCounts = countcats(X7_categorical); % Contar las ocurrencias

% Obtener los nombres únicos de las categorías (días de la semana)
CarsNames = categories(X7_categorical);

% Graficar las frecuencias con bar
figure;
bar(carCounts);
set(gca, 'XTickLabel', CarsNames); % Asignar etiquetas a los ejes X
title('Distribución de Narcas de carros');
xlabel('Marcas de carros');
ylabel('Frecuencia');

%% Matriz de correlación
numVars = [X1 X2 X4 X5 y]; % Solo variables numéricas
corrMatrix = corr(numVars, 'rows', 'pairwise');

% Mostrar matriz
figure; heatmap(corrMatrix, 'Title', 'Matriz de Correlación', ...
    'XLabel', 'Variables', 'YLabel', 'Variables');figure;
subplot(2,2,1); boxplot(X1, y); title('X1 según Y');
subplot(2,2,2); boxplot(X2, y); title('X2 según Y');
subplot(2,2,3); boxplot(X4, y); title('X4 según Y');
subplot(2,2,4); boxplot(X5, y); title('X5 según Y');


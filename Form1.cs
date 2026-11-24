
﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ShashKi
{
    // Структура для представления особи (нейронной сети) - улучшенная версия
    public struct Individual
    {
        public double[][][] weights;     // Веса сети [слой][нейрон][вес]
        public double[][] biases;        // Смещения для каждого слоя
        public double fitness;           // Приспособленность
        public int gamesPlayed;          // Количество сыгранных игр
        public double KingValue; 
        // Метод для глубокого копирования
        public Individual DeepCopy()
        {
            var copy = new Individual();

            // Копирование весов
            copy.weights = new double[weights.Length][][];
            for (int l = 0; l < weights.Length; l++)
            {
                copy.weights[l] = new double[weights[l].Length][];
                for (int i = 0; i < weights[l].Length; i++)
                {
                    copy.weights[l][i] = new double[weights[l][i].Length];
                    Array.Copy(weights[l][i], copy.weights[l][i], weights[l][i].Length);
                }
            }

            // Копирование смещений
            copy.biases = new double[biases.Length][];
            for (int l = 0; l < biases.Length; l++)
            {
                if (biases[l] != null)
                {
                    copy.biases[l] = new double[biases[l].Length];
                    Array.Copy(biases[l], copy.biases[l], biases[l].Length);
                }
            }

            copy.fitness = fitness;
            copy.gamesPlayed = gamesPlayed;
            copy.KingValue = KingValue; 
            return copy;
        }
    }

    public class GeneticCheckers
    {
        // Архитектура сети:
        private readonly int[] layerSizes = { 32, 64, 32, 1 };
        private int populationSize;
        private double mutationRate;
        private double crossoverRate;
        private List<Individual> population;
        private Random rng;
        private int currentGeneration;
        private Individual bestIndividualEver;

        // Константы по ТЗ
        private const double WIN_SCORE = 1.0;
        private const double DRAW_SCORE = 0.0;
        private const double LOSE_SCORE = -2.0;

        // Константы для представления фигур по ТЗ: {-K, -1, 0, 1, K}
        private const double K = 2.0; // дамка
        private const double CHECKER_VALUE = 1.0; // шашка

        public GeneticCheckers(int popSize, double mutRate = 0.0, double crossRate = 0.8)
        {
            populationSize = popSize;
            mutationRate = mutRate;
            crossoverRate = crossRate;
            rng = new Random();
            currentGeneration = 0;
            InitializePopulation();
            bestIndividualEver = population[0];
        }

        // Инициализация популяции
        private void InitializePopulation()
        {
            population = new List<Individual>();
            for (int i = 0; i < populationSize; i++)
            {
                population.Add(CreateRandomIndividual());
            }
        }

        // Создание случайной особи
        private Individual CreateRandomIndividual()
        {
            Individual ind = new Individual();

            // Инициализация весов
            ind.weights = new double[layerSizes.Length - 1][][];
            for (int l = 0; l < layerSizes.Length - 1; l++)
            {
                int neuronsInNextLayer = layerSizes[l + 1];
                int neuronsInCurrentLayer = layerSizes[l];

                ind.weights[l] = new double[neuronsInNextLayer][];
                for (int i = 0; i < neuronsInNextLayer; i++)
                {
                    ind.weights[l][i] = new double[neuronsInCurrentLayer];
                    for (int j = 0; j < neuronsInCurrentLayer; j++)
                    {
                        ind.weights[l][i][j] = (rng.NextDouble() * 0.4) - 0.2;
                    }
                }
            }

            // Инициализация смещений
            ind.biases = new double[layerSizes.Length][];
            for (int l = 1; l < layerSizes.Length; l++)
            {
                ind.biases[l] = new double[layerSizes[l]];
                for (int i = 0; i < layerSizes[l]; i++)
                {
                    ind.biases[l][i] = (rng.NextDouble() * 0.4) - 0.2;
                }
            }

            ind.fitness = 0.0;
            ind.gamesPlayed = 0;
            ind.KingValue = 1.5 + rng.NextDouble() * 2.0; // [1.5 .. 3.5]

            return ind;
        }

        // Преобразование доски во входные данные по ТЗ: {-K, -1, 0, 1, K}
        public double[] BoardToInput(Checker[,] board, bool forWhite, double kingValue)
        {
            double[] input = new double[32];
            int index = 0;

            for (int row = 0; row < 8; row++)
            {
                for (int col = 0; col < 8; col++)
                {
                    if ((row + col) % 2 == 0) // Только черные клетки
                    {
                        var checker = board[row, col];
                        if (checker == null)
                        {
                            input[index] = 0.0;
                        }
                        else if (checker.IsWhite == forWhite)
                        {
                            // Свои фигуры: 1 - шашка, K - дамка
                            input[index] = checker.IsKing ? kingValue : CHECKER_VALUE;
                        }
                        else
                        {
                            // Чужие фигуры: -1 - шашка, -K - дамка
                            input[index] = checker.IsKing ? -kingValue : -CHECKER_VALUE;
                        }
                        index++;
                    }
                }
            }

            return input;
        }

        // Прямое распространение для нейросети
        public double Forward(Individual ind, double[] input)
        {
            if (input.Length != layerSizes[0])
                throw new ArgumentException($"Ожидается {layerSizes[0]} входов, получено {input.Length}");

            double[] activations = input;

            for (int l = 0; l < layerSizes.Length - 1; l++)
            {
                double[] nextActivations = new double[layerSizes[l + 1]];

                for (int i = 0; i < layerSizes[l + 1]; i++)
                {
                    double sum = ind.biases[l + 1][i];

                    for (int j = 0; j < layerSizes[l]; j++)
                    {
                        sum += activations[j] * ind.weights[l][i][j];
                    }

                    // Сигмоидная функция активации
                    nextActivations[i] = 1.0 / (1.0 + Math.Exp(-sum));
                }

                activations = nextActivations;
            }

            return activations[0];
        }

        // ========== УНИВИЦИРОВАННЫЕ МЕТОДЫ РАБОТЫ С ДОСКОЙ ==========

        // Клонирование доски
        public Checker[,] CloneBoard(Checker[,] source)
        {
            var clone = new Checker[8, 8];
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if (source[i, j] != null)
                    {
                        clone[i, j] = new Checker(source[i, j].IsWhite, source[i, j].IsKing);
                    }
                }
            }
            return clone;
        }

        // Проверка позиции
        public bool IsValidPosition(int row, int col)
        {
            return row >= 0 && row < 8 && col >= 0 && col < 8;
        }

        // Выполнение хода на доске
        public void ExecuteMoveOnBoard(Checker[,] board, Move move)
        {
            if (move == null) return;

            Point from = move.From;
            Point to = move.To;

            if (!IsValidPosition(from.Y, from.X) || !IsValidPosition(to.Y, to.X))
                return;

            Checker checker = board[from.Y, from.X];
            if (checker == null) return;

            // 1. Перемещаем шашку
            board[from.Y, from.X] = null;
            board[to.Y, to.X] = checker;

            // 2. Удаляем все взятые шашки
            if (move.IsJump && move.JumpedPieces != null && move.JumpedPieces.Count > 0)
            {
                foreach (Point jumped in move.JumpedPieces)
                {
                    if (IsValidPosition(jumped.Y, jumped.X))
                    {
                        board[jumped.Y, jumped.X] = null;
                    }
                }
            }

            // 3. Превращение в дамку (немедленное)
            if (!checker.IsKing)
            {
                if (checker.IsWhite && to.Y == 0)
                {
                    checker.IsKing = true;
                }
                else if (!checker.IsWhite && to.Y == 7)
                {
                    checker.IsKing = true;
                }
            }
        }

        // Получение всех возможных ходов
        public List<Move> GetAllPossibleMoves(Checker[,] board, bool isWhite)
        {
            var allMoves = new List<Move>();
            var jumpMoves = new List<Move>();

            // Сначала проверяем все возможные взятия
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    var checker = board[i, j];
                    if (checker != null && checker.IsWhite == isWhite)
                    {
                        var checkerJumpMoves = FindJumpMoves(board, i, j, checker, null);
                        if (checkerJumpMoves.Count > 0)
                        {
                            jumpMoves.AddRange(checkerJumpMoves);
                        }
                    }
                }
            }

            // Если есть взятия - возвращаем только их
            if (jumpMoves.Count > 0)
            {
                // Находим максимальную длину цепочки
                int maxJumpLength = jumpMoves.Max(m => m.JumpedPieces.Count);
                if (maxJumpLength > 1)
                {
                    // Если есть цепочки, оставляем только самые длинные
                    return jumpMoves.Where(m => m.JumpedPieces.Count == maxJumpLength).ToList();
                }
                return jumpMoves;
            }

            // Если нет взятий - возвращаем простые ходы
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    var checker = board[i, j];
                    if (checker != null && checker.IsWhite == isWhite)
                    {
                        var checkerMoves = new List<Move>();
                        FindSimpleMoves(board, i, j, checker, checkerMoves);
                        allMoves.AddRange(checkerMoves);
                    }
                }
            }

            return allMoves;
        }

        // Поиск взятий
        private List<Move> FindJumpMoves(Checker[,] board, int row, int col, Checker checker, List<Point> jumpedSoFar)
        {
            if (jumpedSoFar == null)
                jumpedSoFar = new List<Point>();

            var moves = new List<Move>();

            if (!checker.IsKing)
            {
                FindSimpleJumpMoves(board, row, col, checker, jumpedSoFar, moves);
            }
            else
            {
                FindKingJumpMoves(board, row, col, checker, jumpedSoFar, moves);
            }

            return moves;
        }

        // Поиск простых взятий
        private void FindSimpleJumpMoves(Checker[,] board, int row, int col, Checker checker,
            List<Point> jumpedSoFar, List<Move> result)
        {
            int[] rowDirs = { -1, -1, 1, 1 };
            int[] colDirs = { -1, 1, -1, 1 };

            for (int i = 0; i < 4; i++)
            {
                int enemyRow = row + rowDirs[i];
                int enemyCol = col + colDirs[i];
                int landRow = enemyRow + rowDirs[i];
                int landCol = enemyCol + colDirs[i];

                if (!IsValidPosition(enemyRow, enemyCol) ||
                    !IsValidPosition(landRow, landCol))
                    continue;

                if (board[enemyRow, enemyCol] == null ||
                    board[enemyRow, enemyCol].IsWhite == checker.IsWhite ||
                    board[landRow, landCol] != null ||
                    jumpedSoFar.Contains(new Point(enemyCol, enemyRow)))
                    continue;

                var newJumped = new List<Point>(jumpedSoFar)
            {
                new Point(enemyCol, enemyRow)
            };

                var tempBoard = CloneBoard(board);
                var tempChecker = new Checker(checker.IsWhite, checker.IsKing);

                tempBoard[row, col] = null;
                tempBoard[enemyRow, enemyCol] = null;
                tempBoard[landRow, landCol] = tempChecker;

                bool becameKing = !tempChecker.IsKing &&
                    ((tempChecker.IsWhite && landRow == 0) ||
                     (!tempChecker.IsWhite && landRow == 7));

                if (becameKing)
                    tempChecker.IsKing = true;

                var nextMoves = FindJumpMoves(tempBoard, landRow, landCol, tempChecker, newJumped);

                if (nextMoves.Count > 0)
                {
                    foreach (var next in nextMoves)
                    {
                        result.Add(new Move(new Point(col, row), next.To, true, next.JumpedPieces));
                    }
                }
                else
                {
                    result.Add(new Move(new Point(col, row), new Point(landCol, landRow), true, newJumped));
                }
            }
        }

        // Поиск взятий дамками
        private void FindKingJumpMoves(Checker[,] board, int row, int col, Checker checker,
            List<Point> jumpedSoFar, List<Move> result)
        {
            int[] rowDirs = { -1, -1, 1, 1 };
            int[] colDirs = { -1, 1, -1, 1 };

            for (int d = 0; d < 4; d++)
            {
                int r = row + rowDirs[d];
                int c = col + colDirs[d];
                Point? enemy = null;

                while (IsValidPosition(r, c))
                {
                    if (board[r, c] != null)
                    {
                        if (board[r, c].IsWhite != checker.IsWhite &&
                            !jumpedSoFar.Contains(new Point(c, r)))
                        {
                            enemy = new Point(c, r);
                        }
                        break;
                    }
                    r += rowDirs[d];
                    c += colDirs[d];
                }

                if (!enemy.HasValue)
                    continue;

                int landRow = enemy.Value.Y + rowDirs[d];
                int landCol = enemy.Value.X + colDirs[d];

                while (IsValidPosition(landRow, landCol) && board[landRow, landCol] == null)
                {
                    var newJumped = new List<Point>(jumpedSoFar) { enemy.Value };

                    var tempBoard = CloneBoard(board);
                    var tempChecker = new Checker(checker.IsWhite, true);

                    tempBoard[row, col] = null;
                    tempBoard[enemy.Value.Y, enemy.Value.X] = null;
                    tempBoard[landRow, landCol] = tempChecker;

                    var nextMoves = FindJumpMoves(tempBoard, landRow, landCol, tempChecker, newJumped);

                    if (nextMoves.Count > 0)
                    {
                        foreach (var next in nextMoves)
                        {
                            result.Add(new Move(new Point(col, row), next.To, true, next.JumpedPieces));
                        }
                    }
                    else
                    {
                        result.Add(new Move(new Point(col, row), new Point(landCol, landRow), true, newJumped));
                    }

                    landRow += rowDirs[d];
                    landCol += colDirs[d];
                }
            }
        }

        // Поиск простых ходов
        private void FindSimpleMoves(Checker[,] gameBoard, int row, int col, Checker checker, List<Move> moves)
        {
            if (checker.IsKing)
            {
                int[] rowDirs = { -1, -1, 1, 1 };
                int[] colDirs = { -1, 1, -1, 1 };

                for (int i = 0; i < 4; i++)
                {
                    int currentRow = row + rowDirs[i];
                    int currentCol = col + colDirs[i];

                    while (IsValidPosition(currentRow, currentCol))
                    {
                        if (gameBoard[currentRow, currentCol] == null)
                        {
                            moves.Add(new Move(new Point(col, row), new Point(currentCol, currentRow), false));
                        }
                        else
                        {
                            break;
                        }
                        currentRow += rowDirs[i];
                        currentCol += colDirs[i];
                    }
                }
            }
            else
            {
                int moveDirection = checker.IsWhite ? -1 : 1;
                int[] colDirs = { -1, 1 };

                foreach (int colDir in colDirs)
                {
                    int newRow = row + moveDirection;
                    int newCol = col + colDir;

                    if (IsValidPosition(newRow, newCol) && gameBoard[newRow, newCol] == null)
                    {
                        moves.Add(new Move(new Point(col, row), new Point(newCol, newRow), false));
                    }
                }
            }
        }

        // Проверка валидности хода
        private bool IsValidMove(Checker[,] board, Move move, bool isWhite)
        {
            var from = move.From;
            var to = move.To;

            if (!IsValidPosition(from.Y, from.X) || !IsValidPosition(to.Y, to.X))
                return false;

            var checker = board[from.Y, from.X];
            if (checker == null || checker.IsWhite != isWhite)
                return false;

            return true;
        }

        // Проверка победы
        public bool CheckWin(Checker[,] board, bool forWhite)
        {
            // Проверяем, остались ли у противника шашки
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    var checker = board[i, j];
                    if (checker != null && checker.IsWhite == forWhite)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // Проверка ничьи
        private bool CheckDraw(Checker[,] board)
        {
            // Простая проверка на ничью
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if (board[i, j] == null)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // Инициализация доски
        private Checker[,] InitializeBoard()
        {
            var board = new Checker[8, 8];

            // Черные шашки (сверху)
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 8; col++)
                {
                    if ((row + col) % 2 == 1)
                    {
                        board[row, col] = new Checker(false, false);
                    }
                }
            }

            // Белые шашки (снизу)
            for (int row = 5; row < 8; row++)
            {
                for (int col = 0; col < 8; col++)
                {
                    if ((row + col) % 2 == 1)
                    {
                        board[row, col] = new Checker(true, false);
                    }
                }
            }

            return board;
        }

        // ========== МЕТОДЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ==========

        // Получение хода от нейросети с учетом двух шагов вперед
        private Move GetNetworkMove(Checker[,] board, Individual ind, bool isWhite)
        {
            var allMoves = GetAllPossibleMoves(board, isWhite);
            if (allMoves.Count == 0) return null;

            // Сначала проверяем выигрышные ходы
            foreach (var move in allMoves)
            {
                var tempBoard = CloneBoard(board);
                ExecuteMoveOnBoard(tempBoard, move);
                if (CheckWin(tempBoard, !isWhite)) // Противник проиграл
                {
                    return move;
                }
            }

            Move bestMove = null;
            double bestScore = double.MinValue;

            // Для каждого возможного хода просчитываем на два шага вперед
            foreach (var move in allMoves)
            {
                // ШАГ 1: выполняем ход нейросети
                var boardAfterFirst = CloneBoard(board);
                ExecuteMoveOnBoard(boardAfterFirst, move);

                double moveScore = 0;

                // ШАГ 2: просчитываем лучший ответ противника
                var opponentMoves = GetAllPossibleMoves(boardAfterFirst, !isWhite);
                if (opponentMoves.Count > 0)
                {
                    double worstOpponentScore = double.MaxValue;

                    foreach (var opponentMove in opponentMoves)
                    {
                        // Выполняем ход противника
                        var boardAfterSecond = CloneBoard(boardAfterFirst);
                        ExecuteMoveOnBoard(boardAfterSecond, opponentMove);

                        // Оцениваем позицию после двух ходов с использованием KingValue особи
                        var input = BoardToInput(boardAfterSecond, isWhite, ind.KingValue);
                        double score = Forward(ind, input);

                        // Учитываем, что противник будет выбирать худший для нас ход
                        if (score < worstOpponentScore)
                        {
                            worstOpponentScore = score;
                        }
                    }

                    moveScore = worstOpponentScore;
                }
                else
                {
                    // У противника нет ходов - мы выигрываем
                    moveScore = 1000;
                }

                // Бонусы за особенности хода
                if (move.IsJump)
                {
                    moveScore += 0.5; // Бонус за взятие
                    moveScore += move.JumpedPieces.Count * 0.3; // Бонус за каждую взятую шашку
                }

                // Бонус за превращение в дамку
                var checker = board[move.From.Y, move.From.X];
                if (checker != null && !checker.IsKing)
                {
                    if ((checker.IsWhite && move.To.Y == 0) ||
                        (!checker.IsWhite && move.To.Y == 7))
                    {
                        moveScore += 0.8; // Бонус за превращение
                    }
                }

                if (moveScore > bestScore)
                {
                    bestScore = moveScore;
                    bestMove = move;
                }
            }

            return bestMove ?? allMoves[0];
        }

        // Симуляция игры между двумя особями
        // player играет за sideToPlay (true=белые, false=чёрные)
        private int SimulateGame(Individual player, Individual opponent, bool playerPlaysWhite)
        {
            var board = InitializeBoard();
            bool currentPlayerWhite = true; // Белые всегда ходят первыми
            int movesWithoutCapture = 0;
            const int maxMoves = 100;
            const int maxMovesWithoutCapture = 40;

            for (int moveNum = 0; moveNum < maxMoves; moveNum++)
            {
                Move move;

                // Определяем, кто сейчас ходит
                if (currentPlayerWhite)
                {
                    // Белые ходят
                    if (playerPlaysWhite)
                    {
                        move = GetNetworkMove(board, player, currentPlayerWhite);
                    }
                    else
                    {
                        move = GetNetworkMove(board, opponent, currentPlayerWhite);
                    }
                }
                else
                {
                    // Чёрные ходят
                    if (playerPlaysWhite)
                    {
                        move = GetNetworkMove(board, opponent, currentPlayerWhite);
                    }
                    else
                    {
                        move = GetNetworkMove(board, player, currentPlayerWhite);
                    }
                }

                if (move == null || !IsValidMove(board, move, currentPlayerWhite))
                {
                    // Нет валидных ходов - текущий игрок проигрывает
                    // Определяем, проиграл ли player
                    bool playerLost = (currentPlayerWhite && playerPlaysWhite) ||
                                     (!currentPlayerWhite && !playerPlaysWhite);

                    return playerLost ? -1 : 1;
                }

                // Проверяем, было ли взятие
                bool wasCapture = move.IsJump;

                ExecuteMoveOnBoard(board, move);

                if (!wasCapture)
                {
                    movesWithoutCapture++;
                }
                else
                {
                    movesWithoutCapture = 0;
                }

                // Проверка окончания игры
                bool whiteWon = CheckWin(board, false); // Проиграли ли чёрные
                bool blackWon = CheckWin(board, true);  // Проиграли ли белые

                if (whiteWon)
                {
                    // Белые выиграли
                    return playerPlaysWhite ? 1 : -1;
                }
                else if (blackWon)
                {
                    // Чёрные выиграли
                    return playerPlaysWhite ? -1 : 1;
                }

                // Проверка на ничью по правилу 40 ходов без взятия
                if (movesWithoutCapture >= maxMovesWithoutCapture || CheckDraw(board))
                {
                    return 0;
                }

                currentPlayerWhite = !currentPlayerWhite;
            }

            return 0; // Ничья по количеству ходов
        }

        // Вычисление приспособленности по ТЗ
        public void CalculateFitness()
        {
            // Сбрасываем фитнес
            for (int i = 0; i < population.Count; i++)
            {
                var ind = population[i];
                ind.fitness = 0;
                ind.gamesPlayed = 0;
                population[i] = ind;
            }

            // Каждая особь играет с 5 случайными противниками ЗА ЧЁРНЫХ
            for (int i = 0; i < population.Count; i++)
            {
                Individual player = population[i];
                double totalScore = 0;

                for (int g = 0; g < 5; g++)
                {
                    int opponentIndex;
                    do
                    {
                        opponentIndex = rng.Next(population.Count);
                    } while (opponentIndex == i);

                    Individual opponent = population[opponentIndex];

                    // Играем ЗА ЧЁРНЫХ (playerPlaysWhite = false)
                    int result = SimulateGame(player, opponent, false);

                    // Начисляем очки по ТЗ: победа=1, ничья=0, поражение=-2
                    switch (result)
                    {
                        case 1:  // победа чёрных (player)
                            totalScore += WIN_SCORE;
                            break;
                        case 0:  // ничья
                            totalScore += DRAW_SCORE;
                            break;
                        case -1: // поражение чёрных (player)
                            totalScore += LOSE_SCORE;
                            break;
                    }

                    player.gamesPlayed++;
                }

                player.fitness = totalScore;
                population[i] = player;
            }
        }

        // Турнирный отбор
        private Individual TournamentSelection(int tournamentSize = 3)
        {
            Individual best = population[rng.Next(population.Count)];

            for (int i = 1; i < tournamentSize; i++)
            {
                Individual candidate = population[rng.Next(population.Count)];
                if (candidate.fitness > best.fitness)
                {
                    best = candidate;
                }
            }

            return best;
        }

        // Кроссовер 
        private Individual Crossover(Individual parent1, Individual parent2)
        {
            Individual child = parent1.DeepCopy();

            for (int l = 0; l < layerSizes.Length - 1; l++)
            {
                for (int i = 0; i < layerSizes[l + 1]; i++)
                {
                    for (int j = 0; j < layerSizes[l]; j++)
                    {
                        if (rng.NextDouble() < 0.5)
                        {
                            child.weights[l][i][j] = parent2.weights[l][i][j];
                        }
                    }
                }
            }

            // Кроссовер смещений
            for (int l = 1; l < layerSizes.Length; l++)
            {
                for (int i = 0; i < layerSizes[l]; i++)
                {
                    if (rng.NextDouble() < 0.5)
                    {
                        child.biases[l][i] = parent2.biases[l][i];
                    }
                }
            }

            // Кроссовер KingValue (усреднение)
            child.KingValue = (parent1.KingValue + parent2.KingValue) / 2.0;

            // Небольшая мутация KingValue
            if (rng.NextDouble() < 0.3)
            {
                child.KingValue += (rng.NextDouble() - 0.5) * 0.5;
                child.KingValue = Math.Max(1.0, Math.Min(4.0, child.KingValue));
            }

            return child;
        }

        // Мутация 
        private void Mutate(ref Individual ind)
        {
            for (int l = 0; l < layerSizes.Length - 1; l++)
            {
                for (int i = 0; i < layerSizes[l + 1]; i++)
                {
                    for (int j = 0; j < layerSizes[l]; j++)
                    {
                        if (rng.NextDouble() < mutationRate)
                        {
                            ind.weights[l][i][j] += Gaussian() * 0.1;
                            ind.weights[l][i][j] = Math.Max(-1.0, Math.Min(1.0, ind.weights[l][i][j]));
                        }
                    }
                }
            }

            // Мутация смещений
            for (int l = 1; l < layerSizes.Length; l++)
            {
                for (int i = 0; i < layerSizes[l]; i++)
                {
                    if (rng.NextDouble() < mutationRate)
                    {
                        ind.biases[l][i] += Gaussian() * 0.1;
                        ind.biases[l][i] = Math.Max(-1.0, Math.Min(1.0, ind.biases[l][i]));
                    }
                }
            }

            // Мутация KingValue
            if (rng.NextDouble() < mutationRate * 2)
            {
                ind.KingValue += (rng.NextDouble() - 0.5) * 0.3;
                ind.KingValue = Math.Max(1.0, Math.Min(4.0, ind.KingValue));
            }
        }

        private double Gaussian()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        // Эволюция одного поколения ПО ТЗ
        public void Evolve()
        {
            CalculateFitness();

            // Сортируем по фитнесу
            population = population.OrderByDescending(i => i.fitness).ToList();

            // Обновляем лучшую особь
            if (population[0].fitness > bestIndividualEver.fitness)
            {
                bestIndividualEver = population[0].DeepCopy();
            }

            List<Individual> newPopulation = new List<Individual>();

            // 1. 15 лучших переходят в следующее поколение
            int eliteCount = Math.Min(15, population.Count);
            for (int i = 0; i < eliteCount; i++)
            {
                newPopulation.Add(population[i].DeepCopy());
            }

            // 2. Добавляем 15 новых случайных сетей с весами [-0.2, 0.2]
            for (int i = 0; i < 15; i++)
            {
                newPopulation.Add(CreateRandomIndividual());
            }

            // 3. Если нужно больше особей, создаем детей от лучших 15
            while (newPopulation.Count < populationSize)
            {
                // Выбираем двух родителей из лучших 15
                int parent1Index = rng.Next(eliteCount);
                int parent2Index;
                do
                {
                    parent2Index = rng.Next(eliteCount);
                } while (parent2Index == parent1Index);

                Individual parent1 = population[parent1Index];
                Individual parent2 = population[parent2Index];

                // Создаем ребенка через кроссовер
                Individual child = Crossover(parent1, parent2);

                // Мутация
                if (rng.NextDouble() < mutationRate)
                {
                    Mutate(ref child);
                }

                newPopulation.Add(child);
            }

            population = newPopulation;
            currentGeneration++;

            Console.WriteLine($"Поколение {currentGeneration}: " +
                             $"Лучший фитнес = {bestIndividualEver.fitness:F2}, " +
                             $"KingValue = {bestIndividualEver.KingValue:F2}");
        }

        // Обучение
        public void Train(int generations)
        {
            Console.WriteLine("Начало эволюции нейросети для шашек...");
            Console.WriteLine($"Размер популяции: {populationSize}");
            Console.WriteLine($"Мутация: {mutationRate}, Кроссовер: {crossoverRate}");
            Console.WriteLine($"Поколений: {generations}\n");

            for (int gen = 0; gen < generations; gen++)
            {
                Evolve();
            }

            Console.WriteLine("\nОбучение завершено!");
            Console.WriteLine($"Лучшая особь: фитнес = {bestIndividualEver.fitness:F2}, " +
                             $"KingValue = {bestIndividualEver.KingValue:F2}");
        }

        // Получение лучшей особи
        public Individual GetBestIndividual()
        {
            return bestIndividualEver.DeepCopy();
        }
    }

    // ====== ОБНОВЛЕННАЯ ФОРМА ДЛЯ РАБОТЫ С НОВОЙ СТРУКТУРОЙ ======
    public partial class Form1 : Form
    {
        int totalGenerations = 100;//поколений

        private Individual bestNetwork;
        private Checker[,] board = new Checker[8, 8];
        int difficulty = 3;
        private bool isPlayerWhite = true;
        private bool gameStarted = false;
        private Checker selectedChecker = null;
        private List<Point> possibleMoves = new List<Point>();
        private Random random = new Random();
        private GeneticCheckers geneticTrainer;
        private System.Windows.Forms.Timer computerMoveTimer;
        private List<Move> computerMoveChain = new List<Move>();
        private int currentChainIndex = 0;
        

        public Form1()
        {
            InitializeComponent();
            AttachPanelClickHandlers();
            InitializeBoard();
            comboBox1.SelectedIndex = 0;

            // Инициализация генетического алгоритма 
            geneticTrainer = new GeneticCheckers(30);//особей в поколении

            // Инициализация таймера для задержки ходов компьютера
            computerMoveTimer = new System.Windows.Forms.Timer();
            computerMoveTimer.Interval = 800;
            computerMoveTimer.Tick += ComputerMoveTimer_Tick;
        }
        

        private void ComputerMoveTimer_Tick(object sender, EventArgs e)
        {
            computerMoveTimer.Stop();

            if (currentChainIndex < computerMoveChain.Count)
            {
                ExecuteMoveOnBoard(board, computerMoveChain[currentChainIndex]);
                UpdateBoardDisplay();
                currentChainIndex++;

                if (currentChainIndex < computerMoveChain.Count)
                {
                    computerMoveTimer.Start();
                }
                else
                {
                    computerMoveChain.Clear();
                    currentChainIndex = 0;

                    if (CheckWinCondition(board, isPlayerWhite))
                    {
                        MessageBox.Show("Компьютер выиграл!", "Игра окончена",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                        gameStarted = false;
                    }
                }
            }
        }

        private void AttachPanelClickHandlers()
        {
            Panel[] panels =
            {
            p1, p2, p3, p4, p5, p6, p7, p8,
            p9, p10, p11, p12, p13, p14, p15, p16,
            p17, p18, p19, p20, p21, p22, p23, p24,
            p25, p26, p27, p28, p29, p30, p31, p32,
            p33, p34, p35, p36, p37, p38, p39, p40,
            p41, p42, p43, p44, p45, p46, p47, p48,
            p49, p50, p51, p52, p53, p54, p55, p56,
            p57, p58, p59, p60, p61, p62, p63, p64
        };

            for (int i = 0; i < panels.Length; i++)
            {
                int row = i / 8;
                int col = i % 8;
                panels[i].Tag = new Point(col, row);
                panels[i].Click += Panel_Click;
            }
        }

        private void InitializeBoard()
        {
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    board[i, j] = null;

            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 8; col++)
                    if ((row + col) % 2 == 0)
                        board[row, col] = new Checker(false, false);

            for (int row = 5; row < 8; row++)
                for (int col = 0; col < 8; col++)
                    if ((row + col) % 2 == 0)
                        board[row, col] = new Checker(true, false);

            UpdateBoardDisplay();
        }

        // ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ДОСКИ (ИСПОЛЬЗУЮТ GeneticCheckers) ==========

        private List<Move> GetAllPossibleMoves(Checker[,] board, bool isWhite)
        {
            return geneticTrainer.GetAllPossibleMoves(board, isWhite);
        }

        private bool IsValidPosition(int row, int col)
        {
            return geneticTrainer.IsValidPosition(row, col);
        }

        private Checker[,] CloneBoard(Checker[,] sourceBoard = null)
        {
            return geneticTrainer.CloneBoard(sourceBoard ?? board);
        }

        private void ExecuteMoveOnBoard(Checker[,] gameBoard, Move move)
        {
            geneticTrainer.ExecuteMoveOnBoard(gameBoard, move);
        }
        
        /// <summary>
        /// /////////////////////////////////////////////////////////////////////////////////
        /// </summary>
        private double neuralChance = 0.1;  //процент который нейронка играет а не минимакс
        private MinimaxCheckers minimax = new MinimaxCheckers();
        /// <summary>
        /// /////////////////////////////////////////////////////////////////////////////////
        /// </summary>
        /// <param name="gameBoard"></param>
        /// <param name="forWhite"></param>
        /// <returns></returns>
        private bool CheckWinCondition(Checker[,] gameBoard, bool forWhite)
        {
            return geneticTrainer.CheckWin(gameBoard, forWhite);
        }

        // ========== ИГРОВАЯ ЛОГИКА ==========

        private void button2_Click(object sender, EventArgs e)
        {
            button2.Enabled = false;
            progressBar1.Value = 0;

            var worker = new BackgroundWorker { WorkerReportsProgress = true };
            worker.DoWork += (s, args) => TrainNeuralNetwork(worker);
            worker.ProgressChanged += (s, args) => progressBar1.Value = args.ProgressPercentage;
            worker.RunWorkerCompleted += (s, args) =>
            {
                button2.Enabled = true;
                progressBar1.Value = 100;
                MessageBox.Show("Нейросеть успешно обучена!", "Готово",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            };
            worker.RunWorkerAsync();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (bestNetwork.weights == null)
            {
                MessageBox.Show("Сначала обучите нейросеть!", "Ошибка",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            isPlayerWhite = radioButton1.Checked;
            bool playerFirstMove = comboBox1.SelectedItem?.ToString() == "вы";
            gameStarted = true;
            selectedChecker = null;
            possibleMoves.Clear();

            InitializeBoard();
            if (!playerFirstMove) MakeComputerMove();
        }

        private void UpdateBoardDisplay()
        {
            Panel[] panels =
            {
            p1, p2, p3, p4, p5, p6, p7, p8,
            p9, p10, p11, p12, p13, p14, p15, p16,
            p17, p18, p19, p20, p21, p22, p23, p24,
            p25, p26, p27, p28, p29, p30, p31, p32,
            p33, p34, p35, p36, p37, p38, p39, p40,
            p41, p42, p43, p44, p45, p46, p47, p48,
            p49, p50, p51, p52, p53, p54, p55, p56,
            p57, p58, p59, p60, p61, p62, p63, p64
        };

            foreach (var panel in panels)
            {
                Point pos = (Point)panel.Tag;
                int row = pos.Y;
                int col = pos.X;

                var checker = board[row, col];

                panel.BackColor = (row + col) % 2 == 0
                    ? SystemColors.ControlDarkDark
                    : SystemColors.ButtonFace;

                panel.BackgroundImage = null;

                if (checker != null)
                {
                    Color color = checker.IsWhite ? Color.White : Color.Black;
                    panel.BackgroundImage = checker.IsKing
                        ? CreateKingChecker(color)
                        : CreateChecker(color);

                    panel.BackgroundImageLayout = ImageLayout.Center;

                    if (selectedChecker == checker)
                        panel.BackColor = Color.Yellow;
                }

                Point movePoint = new Point(col, row);
                if (possibleMoves.Contains(movePoint))
                {
                    panel.BackColor = IsJumpMove(selectedChecker, movePoint)
                        ? Color.LightCoral
                        : Color.LightGreen;
                }
            }
        }

        private Bitmap CreateChecker(Color color) => CreateCheckerImage(color, false);
        private Bitmap CreateKingChecker(Color color) => CreateCheckerImage(color, true);

        private Bitmap CreateCheckerImage(Color color, bool isKing)
        {
            var bmp = new Bitmap(60, 60);
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.FillEllipse(new SolidBrush(color), 5, 5, 50, 50);
                g.DrawEllipse(new Pen(Color.Gray, 2), 5, 5, 50, 50);

                if (isKing)
                {
                    g.DrawEllipse(new Pen(Color.Gold, 3), 5, 5, 50, 50);
                }
            }
            return bmp;
        }

        private void MakeComputerMove()
        {
            if (!gameStarted || bestNetwork.weights == null) return;

            var moves = GetAllPossibleMoves(board, !isPlayerWhite);

            if (moves.Count == 0)
            {
                MessageBox.Show("У компьютера нет возможных ходов. Вы победили!", "Игра окончена",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                gameStarted = false;
                return;
            }

            Move selectedMove = GetHybridMove();
            if (selectedMove == null) return;

            ExecuteComputerMoveWithVisualization(selectedMove);
        }


        /// <summary>
        /// ////////////////////////////////////////////////////////////////////////////
        /// </summary>
        /// <returns></returns>
        private Move GetHybridMove()
        {
            var moves = GetAllPossibleMoves(board, !isPlayerWhite);
            if (moves.Count == 0)
                return null;

            bool useNeural = random.NextDouble() < neuralChance;
            //return GetNeuralNetworkMove(moves);
            if (totalGenerations <= 10)
            { return GetNeuralNetworkMove(moves); }
                if (useNeural && bestNetwork.weights != null)
            {
                return GetNeuralNetworkMove(moves); //ход леает нейро
            }
            else ///ход делает минимакс, если хочешь просто удали выбор нейро и вставь этот блок вместо if
            {
                return minimax.FindBestMove(
                    board,
                    depth: difficulty,
                    maximizingPlayer: true,
                    computerIsWhite: !isPlayerWhite,
                    getMoves: GetAllPossibleMoves,
                    applyMove: (b, m) =>
                    {
                        var copy = CloneBoard(b);
                        ExecuteMoveOnBoard(copy, m);
                        return copy;
                    },
                    checkWin: CheckWinCondition
                );
            }
        }
        /// <summary>
        /// //////////////////////////////////////////////////////////////////////////////
        /// </summary>
        /// <param name="moves"></param>
        /// <returns></returns>
        private Move GetNeuralNetworkMove(List<Move> moves)
        {
            if (bestNetwork.weights == null || moves.Count == 0)
                return moves[0];

            double bestScore = double.MinValue;
            Move bestMove = moves[0];

            bool computerIsWhite = !isPlayerWhite;

            foreach (var move in moves)
            {
                var tempBoard = CloneBoard(board);
                ExecuteMoveOnBoard(tempBoard, move);

                double[] input = geneticTrainer.BoardToInput(tempBoard, computerIsWhite, bestNetwork.KingValue);
                double score = geneticTrainer.Forward(bestNetwork, input);

                if (move.IsJump)
                {
                    score += move.JumpedPieces.Count * 0.15;

                    var checker = tempBoard[move.To.Y, move.To.X];
                    if (checker != null && !move.IsMultipleJump)
                    {
                        if (!checker.IsKing && ((checker.IsWhite && move.To.Y == 0) ||
                            (!checker.IsWhite && move.To.Y == 7)))
                        {
                            score += 0.3;
                        }
                    }
                }

                if (!move.IsJump)
                {
                    var checker = tempBoard[move.To.Y, move.To.X];
                    if (checker != null && !checker.IsKing)
                    {
                        if (move.To.X == 0 || move.To.X == 7)
                        {
                            score -= 0.1;
                        }

                        if (IsCheckerInDanger(tempBoard, move.To.Y, move.To.X, checker.IsWhite))
                        {
                            score -= 0.2;
                        }
                    }
                }

                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = move;
                }
            }

            Console.WriteLine($"Выбран ход с оценкой: {bestScore:F3}");
            return bestMove;
        }

        private bool IsCheckerInDanger(Checker[,] board, int row, int col, bool isWhite)
        {
            int[] attackDirs = isWhite ? new int[] { 1 } : new int[] { -1 };

            foreach (int dr in attackDirs)
            {
                foreach (int dc in new int[] { -1, 1 })
                {
                    int attackerRow = row - dr;
                    int attackerCol = col - dc;
                    int jumpRow = row + dr;
                    int jumpCol = col + dc;

                    if (IsValidPosition(attackerRow, attackerCol) &&
                        IsValidPosition(jumpRow, jumpCol))
                    {
                        var attacker = board[attackerRow, attackerCol];
                        var jumpCell = board[jumpRow, jumpCol];

                        if (attacker != null && attacker.IsWhite != isWhite &&
                            (attacker.IsKing || (isWhite ? dr == 1 : dr == -1)) &&
                            jumpCell == null)
                        {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        private void ExecuteComputerMoveWithVisualization(Move move)
        {
            if (move.IsJump && move.JumpedPieces.Count > 1)
            {
                ExecuteJumpChain(move);
            }
            else
            {
                ExecuteSingleMoveWithDelay(move);
            }
        }

        private void ExecuteSingleMoveWithDelay(Move move)
        {
            Task.Delay(600).ContinueWith(t =>
            {
                this.Invoke(new Action(() =>
                {
                    ExecuteMoveOnBoard(board, move);
                    UpdateBoardDisplay();

                    var checker = board[move.To.Y, move.To.X];
                    bool becameKing = checker != null && move.IsJump &&
                        ((checker.IsWhite && move.To.Y == 0) || (!checker.IsWhite && move.To.Y == 7));

                    if (becameKing)
                    {
                        Task.Delay(400).ContinueWith(t2 =>
                        {
                            this.Invoke(new Action(() =>
                            {
                                var furtherJumps = FindJumpMoves(board, move.To.Y, move.To.X,
                                    new Checker(checker.IsWhite, true), new List<Point>(move.JumpedPieces));

                                if (furtherJumps.Count > 0)
                                {
                                    ExecuteJumpChain(furtherJumps[0]);
                                }
                                else
                                {
                                    CheckComputerWin();
                                }
                            }));
                        });
                    }
                    else
                    {
                        CheckComputerWin();
                    }
                }));
            });
        }

        private void ExecuteJumpChain(Move firstMove)
        {
            var chain = new List<Move> { firstMove };
            var currentBoard = CloneBoard(board);
            ExecuteMoveOnBoard(currentBoard, firstMove);
            BuildJumpChainRecursive(currentBoard, firstMove, chain);
            ExecuteChainWithDelays(chain, 0);
        }

        private void BuildJumpChainRecursive(Checker[,] currentBoard, Move lastMove, List<Move> chain)
        {
            var checker = currentBoard[lastMove.To.Y, lastMove.To.X];
            if (checker == null) return;

            var furtherJumps = FindJumpMoves(currentBoard, lastMove.To.Y, lastMove.To.X,
                checker, new List<Point>(lastMove.JumpedPieces));

            if (furtherJumps.Count > 0)
            {
                int maxJumps = furtherJumps.Max(m => m.JumpedPieces.Count);
                var bestContinuation = furtherJumps.FirstOrDefault(m => m.JumpedPieces.Count == maxJumps);

                if (bestContinuation != null)
                {
                    chain.Add(bestContinuation);
                    var nextBoard = CloneBoard(currentBoard);
                    ExecuteMoveOnBoard(nextBoard, bestContinuation);
                    BuildJumpChainRecursive(nextBoard, bestContinuation, chain);
                }
            }
        }

        private void ExecuteChainWithDelays(List<Move> chain, int index)
        {
            if (index >= chain.Count)
            {
                CheckComputerWin();
                return;
            }

            Task.Delay(800).ContinueWith(t =>
            {
                this.Invoke(new Action(() =>
                {
                    ExecuteMoveOnBoard(board, chain[index]);
                    UpdateBoardDisplay();
                    ExecuteChainWithDelays(chain, index + 1);
                }));
            });
        }

        private void CheckComputerWin()
        {
            if (CheckWinCondition(board, isPlayerWhite))
            {
                MessageBox.Show("Компьютер выиграл!", "Игра окончена",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                gameStarted = false;
            }
        }

        private void TrainNeuralNetwork(BackgroundWorker worker)
        {
            int progressStep = totalGenerations / 100;

            Console.WriteLine("Начало обучения нейросети...");
            Console.WriteLine($"Всего поколений: {totalGenerations}");

            for (int gen = 0; gen < totalGenerations; gen++)
            {
                geneticTrainer.Train(1);
                bestNetwork = geneticTrainer.GetBestIndividual();

                if (gen % progressStep == 0)
                {
                    worker.ReportProgress(gen * 100 / totalGenerations);
                }
                
                if (worker.CancellationPending)
                {
                    return;
                }
            }

            worker.ReportProgress(100);
            Console.WriteLine("Обучение завершено!");
        }

        // ========== МЕТОДЫ ДЛЯ ПОИСКА ХОДОВ ==========

        private List<Move> FindJumpMoves(Checker[,] board, int row, int col, Checker checker, List<Point> jumpedSoFar = null)
        {
            if (jumpedSoFar == null)
                jumpedSoFar = new List<Point>();

            var moves = new List<Move>();

            if (!checker.IsKing)
            {
                FindSimpleJumpMoves(board, row, col, checker, jumpedSoFar, moves);
            }
            else
            {
                FindKingJumpMoves(board, row, col, checker, jumpedSoFar, moves);
            }

            return moves;
        }

        private void FindSimpleJumpMoves(Checker[,] board, int row, int col, Checker checker,
            List<Point> jumpedSoFar, List<Move> result)
        {
            int[] rowDirs = { -1, -1, 1, 1 };
            int[] colDirs = { -1, 1, -1, 1 };

            for (int i = 0; i < 4; i++)
            {
                int enemyRow = row + rowDirs[i];
                int enemyCol = col + colDirs[i];
                int landRow = enemyRow + rowDirs[i];
                int landCol = enemyCol + colDirs[i];

                if (!IsValidPosition(enemyRow, enemyCol) ||
                    !IsValidPosition(landRow, landCol))
                    continue;

                if (board[enemyRow, enemyCol] == null ||
                    board[enemyRow, enemyCol].IsWhite == checker.IsWhite ||
                    board[landRow, landCol] != null ||
                    jumpedSoFar.Contains(new Point(enemyCol, enemyRow)))
                    continue;

                var newJumped = new List<Point>(jumpedSoFar)
            {
                new Point(enemyCol, enemyRow)
            };

                var tempBoard = CloneBoard(board);
                var tempChecker = new Checker(checker.IsWhite, checker.IsKing);

                tempBoard[row, col] = null;
                tempBoard[enemyRow, enemyCol] = null;
                tempBoard[landRow, landCol] = tempChecker;

                bool becameKing = !tempChecker.IsKing &&
                    ((tempChecker.IsWhite && landRow == 0) ||
                     (!tempChecker.IsWhite && landRow == 7));

                if (becameKing)
                    tempChecker.IsKing = true;

                var nextMoves = FindJumpMoves(tempBoard, landRow, landCol, tempChecker, newJumped);

                if (nextMoves.Count > 0)
                {
                    foreach (var next in nextMoves)
                    {
                        result.Add(new Move(new Point(col, row), next.To, true, next.JumpedPieces));
                    }
                }
                else
                {
                    result.Add(new Move(new Point(col, row), new Point(landCol, landRow), true, newJumped));
                }
            }
        }

        private void FindKingJumpMoves(Checker[,] board, int row, int col, Checker checker,
            List<Point> jumpedSoFar, List<Move> result)
        {
            int[] rowDirs = { -1, -1, 1, 1 };
            int[] colDirs = { -1, 1, -1, 1 };

            for (int d = 0; d < 4; d++)
            {
                int r = row + rowDirs[d];
                int c = col + colDirs[d];
                Point? enemy = null;

                while (IsValidPosition(r, c))
                {
                    if (board[r, c] != null)
                    {
                        if (board[r, c].IsWhite != checker.IsWhite &&
                            !jumpedSoFar.Contains(new Point(c, r)))
                        {
                            enemy = new Point(c, r);
                        }
                        break;
                    }
                    r += rowDirs[d];
                    c += colDirs[d];
                }

                if (!enemy.HasValue)
                    continue;

                int landRow = enemy.Value.Y + rowDirs[d];
                int landCol = enemy.Value.X + colDirs[d];

                while (IsValidPosition(landRow, landCol) && board[landRow, landCol] == null)
                {
                    var newJumped = new List<Point>(jumpedSoFar) { enemy.Value };

                    var tempBoard = CloneBoard(board);
                    var tempChecker = new Checker(checker.IsWhite, true);

                    tempBoard[row, col] = null;
                    tempBoard[enemy.Value.Y, enemy.Value.X] = null;
                    tempBoard[landRow, landCol] = tempChecker;

                    var nextMoves = FindJumpMoves(tempBoard, landRow, landCol, tempChecker, newJumped);

                    if (nextMoves.Count > 0)
                    {
                        foreach (var next in nextMoves)
                        {
                            result.Add(new Move(new Point(col, row), next.To, true, next.JumpedPieces));
                        }
                    }
                    else
                    {
                        result.Add(new Move(new Point(col, row), new Point(landCol, landRow), true, newJumped));
                    }

                    landRow += rowDirs[d];
                    landCol += colDirs[d];
                }
            }
        }

        private void FindSimpleMoves(Checker[,] gameBoard, int row, int col, Checker checker, List<Move> moves)
        {
            if (checker.IsKing)
            {
                int[] rowDirs = { -1, -1, 1, 1 };
                int[] colDirs = { -1, 1, -1, 1 };

                for (int i = 0; i < 4; i++)
                {
                    int currentRow = row + rowDirs[i];
                    int currentCol = col + colDirs[i];

                    while (IsValidPosition(currentRow, currentCol))
                    {
                        if (gameBoard[currentRow, currentCol] == null)
                        {
                            moves.Add(new Move(new Point(col, row), new Point(currentCol, currentRow), false));
                        }
                        else
                        {
                            break;
                        }
                        currentRow += rowDirs[i];
                        currentCol += colDirs[i];
                    }
                }
            }
            else
            {
                int moveDirection = checker.IsWhite ? -1 : 1;
                int[] colDirs = { -1, 1 };

                foreach (int colDir in colDirs)
                {
                    int newRow = row + moveDirection;
                    int newCol = col + colDir;

                    if (IsValidPosition(newRow, newCol) && gameBoard[newRow, newCol] == null)
                    {
                        moves.Add(new Move(new Point(col, row), new Point(newCol, newRow), false));
                    }
                }
            }
        }

        // ========== ЛОГИКА ИГРОКА ==========

        private void Panel_Click(object sender, EventArgs e)
        {
            if (!gameStarted) return;

            Panel panel = sender as Panel;
            if (panel?.Tag == null) return;

            Point pos = (Point)panel.Tag;
            int row = pos.Y;
            int col = pos.X;

            HandleBoardClick(row, col);
        }

        private void HandleBoardClick(int row, int col)
        {
            var clickedChecker = board[row, col];

            if (clickedChecker != null && clickedChecker.IsWhite == isPlayerWhite)
            {
                var allPossibleMoves = GetAllPossibleMoves(board, isPlayerWhite);
                bool hasMandatoryJumps = allPossibleMoves.Any(m => m.IsJump);

                if (hasMandatoryJumps)
                {
                    var checkerMoves = GetAllPossibleMovesForChecker(row, col, board, isPlayerWhite);
                    var mandatoryJumpsForThisChecker = checkerMoves.Where(m => m.IsJump).ToList();

                    if (mandatoryJumpsForThisChecker.Count > 0)
                    {
                        selectedChecker = clickedChecker;
                        possibleMoves.Clear();
                        foreach (var move in mandatoryJumpsForThisChecker)
                        {
                            possibleMoves.Add(move.To);
                        }
                        UpdateBoardDisplay();
                        return;
                    }
                    else
                    {
                        MessageBox.Show("Вы должны выбрать шашку, которая может взять противника!",
                            "Обязательное взятие", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                }
                else
                {
                    selectedChecker = clickedChecker;
                    possibleMoves.Clear();
                    FindPossibleMoves(row, col);
                    UpdateBoardDisplay();
                }
            }
            else if (selectedChecker != null)
            {
                var targetPoint = new Point(col, row);
                if (possibleMoves.Contains(targetPoint))
                {
                    int fromRow = -1, fromCol = -1;

                    for (int r = 0; r < 8; r++)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            if (board[r, c] == selectedChecker)
                            {
                                fromRow = r;
                                fromCol = c;
                                break;
                            }
                        }
                        if (fromRow != -1) break;
                    }

                    if (fromRow != -1)
                    {
                        ExecutePlayerMove(fromRow, fromCol, row, col);
                    }
                }
            }
        }

        private List<Move> GetAllPossibleMovesForChecker(int row, int col, Checker[,] board, bool isWhite)
        {
            var checker = board[row, col];
            if (checker == null || checker.IsWhite != isWhite) return new List<Move>();

            var moves = new List<Move>();

            var jumpMoves = FindJumpMoves(board, row, col, checker);
            if (jumpMoves.Count > 0)
            {
                int maxJumpLength = jumpMoves.Max(m => m.JumpedPieces.Count);
                return jumpMoves.Where(m => m.JumpedPieces.Count == maxJumpLength).ToList();
            }

            FindSimpleMoves(board, row, col, checker, moves);
            return moves;
        }

        private void FindPossibleMoves(int row, int col)
        {
            possibleMoves.Clear();
            var checker = board[row, col];
            if (checker == null) return;

            var allMoves = GetAllPossibleMovesForChecker(row, col, board, isPlayerWhite);

            foreach (var move in allMoves)
            {
                possibleMoves.Add(move.To);
            }
        }

        private void ExecutePlayerMove(int fromRow, int fromCol, int toRow, int toCol)
        {
            var checker = board[fromRow, fromCol];
            if (checker == null) return;

            var allMoves = GetAllPossibleMovesForChecker(fromRow, fromCol, board, isPlayerWhite);
            var move = allMoves.FirstOrDefault(m => m.To == new Point(toCol, toRow));

            if (move == null) return;

            bool wasKingBefore = checker.IsKing;

            ExecuteMoveOnBoard(board, move);
            UpdateBoardDisplay();

            var movedChecker = board[toRow, toCol];
            bool becameKing = movedChecker != null && !wasKingBefore && movedChecker.IsKing;

            if (becameKing)
            {
                MessageBox.Show("Шашка стала дамкой!", "Превращение",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);

                UpdateBoardDisplay();
                Task.Delay(500).Wait();
            }

            if (move.IsJump && movedChecker != null)
            {
                var furtherJumps = FindJumpMoves(board, toRow, toCol,
                    movedChecker, new List<Point>(move.JumpedPieces));

                if (furtherJumps.Count > 0)
                {
                    selectedChecker = movedChecker;
                    possibleMoves.Clear();

                    foreach (var furtherMove in furtherJumps)
                        possibleMoves.Add(furtherMove.To);

                    UpdateBoardDisplay();

                    string message = "Вы можете продолжить взятие!";
                    if (becameKing)
                        message += "\nДамка может продолжать взятие в любом направлении!";

                    MessageBox.Show(message, "Продолжение",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);

                    return;
                }
            }

            selectedChecker = null;
            possibleMoves.Clear();
            UpdateBoardDisplay();

            if (CheckWinCondition(board, !isPlayerWhite))
            {
                MessageBox.Show("Вы выиграли!", "Игра окончена",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);

                gameStarted = false;
                return;
            }

            MakeComputerMove();
        }

        private Point FindCheckerPosition(Checker checker)
        {
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    if (board[i, j] == checker)
                        return new Point(j, i);
            return Point.Empty;
        }

        private bool IsJumpMove(Checker checker, Point targetPoint)
        {
            if (checker == null) return false;

            var currentPos = FindCheckerPosition(checker);
            var allMoves = new List<Move>();
            FindMovesForChecker(board, currentPos.Y, currentPos.X, allMoves);

            return allMoves.Any(m => m.To == targetPoint && m.IsJump);
        }

        private void FindMovesForChecker(Checker[,] board, int row, int col, List<Move> moves,
            bool mustJump = false, List<Point> jumpedSoFar = null)
        {
            var checker = board[row, col];
            if (checker == null) return;

            var jumpMoves = FindJumpMoves(board, row, col, checker, jumpedSoFar);

            if (mustJump && jumpMoves.Count == 0)
                return;

            if (jumpMoves.Count > 0)
            {
                moves.AddRange(jumpMoves);
                return;
            }

            if (!mustJump)
            {
                FindSimpleMoves(board, row, col, checker, moves);
            }
        }
    }

    public class Checker
    {
        public bool IsWhite { get; set; }
        public bool IsKing { get; set; }

        public Checker(bool isWhite, bool isKing)
        {
            IsWhite = isWhite;
            IsKing = isKing;
        }
    }
    public class MinimaxCheckers
    {
        private const double WIN = 10000;
        private const double LOSE = -10000;

        public Move FindBestMove(
            Checker[,] board,
            int depth,
            bool maximizingPlayer,
            bool computerIsWhite,
            Func<Checker[,], bool, List<Move>> getMoves,
            Func<Checker[,], Move, Checker[,]> applyMove,
            Func<Checker[,], bool, bool> checkWin)
        {
            double bestValue = double.NegativeInfinity;
            Move bestMove = null;

            var moves = getMoves(board, computerIsWhite);

            foreach (var move in moves)
            {
                var newBoard = applyMove(board, move);
                double value = Minimax(
                    newBoard,
                    depth - 1,
                    false,
                    computerIsWhite,
                    double.NegativeInfinity,
                    double.PositiveInfinity,
                    getMoves,
                    applyMove,
                    checkWin
                );

                if (value > bestValue)
                {
                    bestValue = value;
                    bestMove = move;
                }
            }

            return bestMove;
        }

        private double Minimax(
            Checker[,] board,
            int depth,
            bool maximizing,
            bool computerIsWhite,
            double alpha,
            double beta,
            Func<Checker[,], bool, List<Move>> getMoves,
            Func<Checker[,], Move, Checker[,]> applyMove,
            Func<Checker[,], bool, bool> checkWin)
        {
            if (depth == 0)
                return Evaluate(board, computerIsWhite);

            if (checkWin(board, computerIsWhite))
                return WIN;

            if (checkWin(board, !computerIsWhite))
                return LOSE;

            bool currentSide = maximizing ? computerIsWhite : !computerIsWhite;
            var moves = getMoves(board, currentSide);

            if (moves.Count == 0)
                return maximizing ? LOSE : WIN;

            if (maximizing)
            {
                double maxEval = double.NegativeInfinity;
                foreach (var move in moves)
                {
                    var newBoard = applyMove(board, move);
                    double eval = Minimax(newBoard, depth - 1, false, computerIsWhite,
                        alpha, beta, getMoves, applyMove, checkWin);
                    maxEval = Math.Max(maxEval, eval);
                    alpha = Math.Max(alpha, eval);
                    if (beta <= alpha) break;
                }
                return maxEval;
            }
            else
            {
                double minEval = double.PositiveInfinity;
                foreach (var move in moves)
                {
                    var newBoard = applyMove(board, move);
                    double eval = Minimax(newBoard, depth - 1, true, computerIsWhite,
                        alpha, beta, getMoves, applyMove, checkWin);
                    minEval = Math.Min(minEval, eval);
                    beta = Math.Min(beta, eval);
                    if (beta <= alpha) break;
                }
                return minEval;
            }
        }

        private double Evaluate(Checker[,] board, bool computerIsWhite)
        {
            double score = 0;

            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    var ch = board[r, c];
                    if (ch == null) continue;

                    double value = ch.IsKing ? 5 : 3;

                    if (ch.IsWhite == computerIsWhite)
                        score += value;
                    else
                        score -= value;
                }
            }

            return score;
        }
    }

    public class Move
    {
        public Point From { get; set; }
        public Point To { get; set; }
        public bool IsJump { get; set; }
        public List<Point> JumpedPieces { get; set; } // Для хранения всех взятых шашек
        public bool IsMultipleJump { get { return JumpedPieces != null && JumpedPieces.Count > 1; } }

        public Move(Point from, Point to, bool isJump)
        {
            From = from;
            To = to;
            IsJump = isJump;
            JumpedPieces = new List<Point>();
        }

        public Move(Point from, Point to, bool isJump, List<Point> jumped)
        {
            From = from;
            To = to;
            IsJump = isJump;
            JumpedPieces = jumped != null
            ? new List<Point>(jumped)   // 🔥 КЛЮЧЕВО
            : new List<Point>();
        }
    }
}

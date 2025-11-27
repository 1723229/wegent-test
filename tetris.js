class Tetris {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.nextCanvas = document.getElementById('next-canvas');
        this.nextCtx = this.nextCanvas.getContext('2d');

        this.COLS = 10;
        this.ROWS = 20;
        this.BLOCK_SIZE = 30;

        this.board = this.createBoard();
        this.currentPiece = null;
        this.nextPiece = null;
        this.score = 0;
        this.level = 1;
        this.lines = 0;
        this.speed = 1000;
        this.gameRunning = false;
        this.gamePaused = false;
        this.gameOver = false;
        this.dropTime = 0;

        this.initPieces();
        this.bindEvents();
        this.updateDisplay();
    }

    createBoard() {
        return Array(this.ROWS).fill().map(() => Array(this.COLS).fill(0));
    }

    initPieces() {
        this.pieces = [
            {
                shape: [
                    [1, 1, 1, 1]
                ],
                color: '#00f5ff'
            },
            {
                shape: [
                    [1, 1],
                    [1, 1]
                ],
                color: '#ffff00'
            },
            {
                shape: [
                    [0, 1, 0],
                    [1, 1, 1]
                ],
                color: '#800080'
            },
            {
                shape: [
                    [0, 1, 1],
                    [1, 1, 0]
                ],
                color: '#00ff00'
            },
            {
                shape: [
                    [1, 1, 0],
                    [0, 1, 1]
                ],
                color: '#ff0000'
            },
            {
                shape: [
                    [1, 0, 0],
                    [1, 1, 1]
                ],
                color: '#ff7f00'
            },
            {
                shape: [
                    [0, 0, 1],
                    [1, 1, 1]
                ],
                color: '#0000ff'
            }
        ];
    }

    createPiece() {
        const pieceIndex = Math.floor(Math.random() * this.pieces.length);
        const piece = this.pieces[pieceIndex];
        return {
            shape: piece.shape.map(row => [...row]),
            color: piece.color,
            x: Math.floor(this.COLS / 2) - Math.floor(piece.shape[0].length / 2),
            y: 0
        };
    }

    rotatePiece(piece) {
        const rotated = [];
        const rows = piece.shape.length;
        const cols = piece.shape[0].length;

        for (let i = 0; i < cols; i++) {
            rotated[i] = [];
            for (let j = rows - 1; j >= 0; j--) {
                rotated[i][rows - 1 - j] = piece.shape[j][i];
            }
        }

        return {
            ...piece,
            shape: rotated
        };
    }

    isValidMove(piece, dx, dy) {
        for (let row = 0; row < piece.shape.length; row++) {
            for (let col = 0; col < piece.shape[row].length; col++) {
                if (piece.shape[row][col]) {
                    const newX = piece.x + col + dx;
                    const newY = piece.y + row + dy;

                    if (newX < 0 || newX >= this.COLS || newY >= this.ROWS) {
                        return false;
                    }

                    if (newY >= 0 && this.board[newY][newX]) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    movePiece(dx, dy) {
        if (this.currentPiece && this.isValidMove(this.currentPiece, dx, dy)) {
            this.currentPiece.x += dx;
            this.currentPiece.y += dy;
            return true;
        }
        return false;
    }

    rotate() {
        if (this.currentPiece) {
            const rotated = this.rotatePiece(this.currentPiece);
            if (this.isValidMove(rotated, 0, 0)) {
                this.currentPiece = rotated;
            }
        }
    }

    dropPiece() {
        if (this.currentPiece) {
            if (!this.movePiece(0, 1)) {
                this.lockPiece();
                this.clearLines();
                this.spawnPiece();
            }
        }
    }

    hardDrop() {
        if (this.currentPiece) {
            while (this.movePiece(0, 1)) {}
            this.dropPiece();
        }
    }

    lockPiece() {
        if (this.currentPiece) {
            for (let row = 0; row < this.currentPiece.shape.length; row++) {
                for (let col = 0; col < this.currentPiece.shape[row].length; col++) {
                    if (this.currentPiece.shape[row][col]) {
                        const y = this.currentPiece.y + row;
                        const x = this.currentPiece.x + col;
                        if (y >= 0) {
                            this.board[y][x] = this.currentPiece.color;
                        }
                    }
                }
            }
        }
    }

    clearLines() {
        let linesCleared = 0;
        const linesToClear = [];

        for (let row = 0; row < this.ROWS; row++) {
            if (this.board[row].every(cell => cell !== 0)) {
                linesToClear.push(row);
            }
        }

        if (linesToClear.length > 0) {
            linesToClear.reverse().forEach(row => {
                this.board.splice(row, 1);
                this.board.unshift(Array(this.COLS).fill(0));
            });

            linesCleared = linesToClear.length;
            this.lines += linesCleared;
            this.score += this.calculateScore(linesCleared);
            this.level = Math.floor(this.lines / 10) + 1;
            this.speed = Math.max(100, 1000 - (this.level - 1) * 100);
            this.updateDisplay();
        }
    }

    calculateScore(lines) {
        const baseScores = [0, 100, 300, 500, 800];
        return baseScores[lines] * this.level;
    }

    spawnPiece() {
        this.currentPiece = this.nextPiece || this.createPiece();
        this.nextPiece = this.createPiece();

        if (!this.isValidMove(this.currentPiece, 0, 0)) {
            this.gameOver = true;
            this.gameRunning = false;
            this.showGameOver();
        }
    }

    showGameOver() {
        document.getElementById('final-score').textContent = this.score;
        document.getElementById('game-over').classList.remove('hidden');
    }

    hideGameOver() {
        document.getElementById('game-over').classList.add('hidden');
    }

    updateDisplay() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('level').textContent = this.level;
        document.getElementById('lines').textContent = this.lines;
    }

    draw() {
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawBoard();
        this.drawPiece(this.currentPiece);

        this.nextCtx.fillStyle = '#f7fafc';
        this.nextCtx.fillRect(0, 0, this.nextCanvas.width, this.nextCanvas.height);

        if (this.nextPiece) {
            this.drawNextPiece();
        }
    }

    drawBoard() {
        for (let row = 0; row < this.ROWS; row++) {
            for (let col = 0; col < this.COLS; col++) {
                if (this.board[row][col]) {
                    this.ctx.fillStyle = this.board[row][col];
                    this.ctx.fillRect(
                        col * this.BLOCK_SIZE,
                        row * this.BLOCK_SIZE,
                        this.BLOCK_SIZE - 1,
                        this.BLOCK_SIZE - 1
                    );

                    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                    this.ctx.fillRect(
                        col * this.BLOCK_SIZE + 2,
                        row * this.BLOCK_SIZE + 2,
                        this.BLOCK_SIZE - 5,
                        this.BLOCK_SIZE - 5
                    );
                }
            }
        }
    }

    drawPiece(piece) {
        if (piece) {
            for (let row = 0; row < piece.shape.length; row++) {
                for (let col = 0; col < piece.shape[row].length; col++) {
                    if (piece.shape[row][col]) {
                        this.ctx.fillStyle = piece.color;
                        this.ctx.fillRect(
                            (piece.x + col) * this.BLOCK_SIZE,
                            (piece.y + row) * this.BLOCK_SIZE,
                            this.BLOCK_SIZE - 1,
                            this.BLOCK_SIZE - 1
                        );

                        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                        this.ctx.fillRect(
                            (piece.x + col) * this.BLOCK_SIZE + 2,
                            (piece.y + row) * this.BLOCK_SIZE + 2,
                            this.BLOCK_SIZE - 5,
                            this.BLOCK_SIZE - 5
                        );
                    }
                }
            }
        }
    }

    drawNextPiece() {
        const blockSize = 20;
        const offsetX = (this.nextCanvas.width - this.nextPiece.shape[0].length * blockSize) / 2;
        const offsetY = (this.nextCanvas.height - this.nextPiece.shape.length * blockSize) / 2;

        for (let row = 0; row < this.nextPiece.shape.length; row++) {
            for (let col = 0; col < this.nextPiece.shape[row].length; col++) {
                if (this.nextPiece.shape[row][col]) {
                    this.nextCtx.fillStyle = this.nextPiece.color;
                    this.nextCtx.fillRect(
                        offsetX + col * blockSize,
                        offsetY + row * blockSize,
                        blockSize - 1,
                        blockSize - 1
                    );

                    this.nextCtx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                    this.nextCtx.fillRect(
                        offsetX + col * blockSize + 1,
                        offsetY + row * blockSize + 1,
                        blockSize - 3,
                        blockSize - 3
                    );
                }
            }
        }
    }

    bindEvents() {
        document.addEventListener('keydown', (e) => {
            if (!this.gameRunning || this.gamePaused) return;

            switch (e.code) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.movePiece(-1, 0);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.movePiece(1, 0);
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    this.dropPiece();
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this.rotate();
                    break;
                case 'Space':
                    e.preventDefault();
                    this.hardDrop();
                    break;
                case 'KeyP':
                    this.togglePause();
                    break;
            }
        });

        document.getElementById('start-btn').addEventListener('click', () => this.start());
        document.getElementById('pause-btn').addEventListener('click', () => this.togglePause());
        document.getElementById('reset-btn').addEventListener('click', () => this.reset());
        document.getElementById('restart-btn').addEventListener('click', () => {
            this.hideGameOver();
            this.reset();
        });
    }

    start() {
        if (!this.gameRunning) {
            this.gameRunning = true;
            this.gamePaused = false;
            this.gameOver = false;
            this.board = this.createBoard();
            this.score = 0;
            this.level = 1;
            this.lines = 0;
            this.speed = 1000;
            this.spawnPiece();
            this.updateDisplay();
            this.gameLoop();
        }
    }

    pause() {
        this.gamePaused = true;
    }

    resume() {
        this.gamePaused = false;
    }

    togglePause() {
        if (this.gameRunning) {
            if (this.gamePaused) {
                this.resume();
            } else {
                this.pause();
            }
        }
    }

    reset() {
        this.gameRunning = false;
        this.gamePaused = false;
        this.gameOver = false;
        this.board = this.createBoard();
        this.currentPiece = null;
        this.nextPiece = null;
        this.score = 0;
        this.level = 1;
        this.lines = 0;
        this.speed = 1000;
        this.updateDisplay();
        this.draw();
        this.hideGameOver();
    }

    gameLoop(currentTime = 0) {
        if (this.gameRunning && !this.gamePaused) {
            if (currentTime - this.dropTime >= this.speed) {
                this.dropPiece();
                this.dropTime = currentTime;
            }

            this.draw();
            requestAnimationFrame((time) => this.gameLoop(time));
        } else if (this.gameRunning && this.gamePaused) {
            this.drawPauseScreen();
            requestAnimationFrame((time) => this.gameLoop(time));
        }
    }

    drawPauseScreen() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = 'white';
        this.ctx.font = '48px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('PAUSED', this.canvas.width / 2, this.canvas.height / 2);

        this.ctx.font = '20px Arial';
        this.ctx.fillText('按 P 继续游戏', this.canvas.width / 2, this.canvas.height / 2 + 40);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new Tetris();
});
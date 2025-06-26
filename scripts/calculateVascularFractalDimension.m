function [fractal_dimension, log_x, log_y] = calculateVascularFractalDimension(image, blockSizeDisplay, blockPlotOn, graphPlotOn)
% Computes box‐counting fractal dimension on a binary image.

% Check if image is binary
if numel(unique(image)) > 2
    error('Input image must be a binary image.');
end

% Initialize variables for storing results
log_x = [];
log_y = [];

% Minimum and maximum division factors
division_factors = [3, 6, 12, 24, 48, 96, 192, 384];

% Image dimensions
[rows, cols] = size(image);

for n = division_factors
    % Calculate box size (s) and its reciprocal
    box_size = rows / n;  % rows = 768
    reciprocal_box_size = 1 / box_size;

    % Determine the number of blocks along each dimension
    block_size_row = floor(rows / n);
    block_size_col = floor(cols / n);

    % Initialize counter for blocks with vessels
    count_blocks_with_vessels = 0;

    for i = 0:n-1
        for j = 0:n-1

            % Calculate block limits with edge handling
            row_start = i*block_size_row + 1;
            row_end = min((i+1)*block_size_row, rows);
            col_start = j*block_size_col + 1;
            col_end = min((j+1)*block_size_col, cols);

            % Extract the block
            block = image(row_start:row_end, col_start:col_end);

            % Check if any vessels in the block
            if any(block(:))
                count_blocks_with_vessels = count_blocks_with_vessels + 1;
            end
        end
    end

    % Store log(1/s) vs log(N(s))
    log_x = [log_x; log10(reciprocal_box_size)];  % log(1/s)
    log_y = [log_y; log10(count_blocks_with_vessels)];  % log(N(s))
end

% Perform linear regression
coeffs = polyfit(log_x, log_y, 1);
fractal_dimension = coeffs(1); % The slope of the regression line is the fractal dimension

% Optionally show Fractal Dimension graph
if graphPlotOn
    figure;
    scatter(log_x, log_y, 'b', 'filled');
    hold on;
    x_fit = linspace(min(log_x), max(log_x), 100);
    y_fit = polyval(coeffs, x_fit);
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
    set(gca,'fontname','Helvetica','FontSize',16);
    xlabel('log(1/box size)');  % Corrected axis label
    ylabel('log(Boxes with vessels)');
    title('Regression Analysis for Vascular Fractal Dimension');
    legend('Data Points', 'Fit Line', 'Location', 'best');
    grid on;
    set(gcf,'color','w');
    hold off;
end

% Optionally show the image with blockSize * blockSize bounding boxes
if blockPlotOn
    if blockSizeDisplay > 0
        num_blocks = blockSizeDisplay;
        block_size_row = floor(rows / num_blocks);
        block_size_col = floor(cols / num_blocks);
        figure;
        imshow(image);
        hold on;
        for i = 0:num_blocks-1
            for j = 0:num_blocks-1
                row_start = i*block_size_row + 1;
                col_start = j*block_size_col + 1;
                rectangle('Position', [col_start, row_start, block_size_col, block_size_row], 'EdgeColor', 'g');
            end
        end
        set(gcf,'color','w');
        hold off;
    end
end

end
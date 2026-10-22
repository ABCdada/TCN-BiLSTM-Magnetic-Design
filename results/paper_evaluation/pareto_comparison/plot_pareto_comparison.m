clear;
close all;
clc;

% Pareto comparison of five transient hysteresis models.
% The x coordinate is the arithmetic mean of the trainable-parameter
% counts reported for Materials A--E. Lower values on both axes are better.

outDir = fileparts(mfilename('fullpath'));

modelNames = { ...
    'Physics-mechanical [13]', ...
    'LSTM-DAO [14]', ...
    'MagLearn 2 [15]', ...
    'GRU-Tiny [16]', ...
    'TCN-BiLSTM (this work)'};

% Material-specific parameter counts reported by the corresponding papers.
% MagLearn 2 and GRU-Tiny use one reported parameter count for all materials.
parameterCounts = { ...
    [87, 81, 78, 90, 81], ...
    [8600, 8600, 4186, 5258, 6176], ...
    860929 * ones(1, 5), ...
    325 * ones(1, 5), ...
    [497793, 2092545, 2092545, 2512593, 2092545]};

meanParameters = cellfun(@mean, parameterCounts);

% Five-material averages of the reported 95th-percentile errors.
meanSeqP95 = mean([ ...
    34.97, 20.07, 25.18, 39.61, 19.69; ...
    12.67,  5.591, 13.07,  6.209, 8.447; ...
    13.99,  4.32,  11.13,  4.94,  8.51; ...
    22.99, 17.46,  27.63, 16.20, 18.38; ...
    12.98,  4.30,  11.09,  5.40,  5.41], 2)';

% MagLearn 2 does not report material-wise energy P95 values in the source
% used for the sequence comparison; it is therefore shown as N/A rather
% than assigning an unsupported numerical value.
meanEnergyP95 = [ ...
    mean([8.61, 4.14, 11.72, 14.67, 3.44]), ...
    mean([1.002, 1.37, 3.665, 1.918, 0.7585]), ...
    NaN, ...
    mean([2.03, 3.96, 4.91, 4.18, 1.73]), ...
    mean([0.96, 0.79, 3.49, 1.32, 0.41])];

colors = [ ...
    0.20, 0.45, 0.70; ...
    0.16, 0.62, 0.48; ...
    0.93, 0.69, 0.13; ...
    0.48, 0.33, 0.65; ...
    0.80, 0.20, 0.18];
markers = {'o', 's', '^', 'd', 'p'};

makeParetoFigure(meanParameters, meanSeqP95, modelNames, colors, markers, ...
    'Average 95th-percentile sequence error (%)', ...
    'pareto_sequence_p95', false, outDir);

makeParetoFigure(meanParameters, meanEnergyP95, modelNames, colors, markers, ...
    'Average 95th-percentile energy error (%)', ...
    'pareto_energy_p95', true, outDir);

fprintf('Average parameter counts used for the x-axis:\n');
for i = 1:numel(modelNames)
    fprintf('  %-28s %.1f\n', modelNames{i}, meanParameters(i));
end
fprintf('\nAverage sequence P95 errors (%%):\n');
disp(meanSeqP95);
fprintf('Average energy P95 errors (%%):\n');
disp(meanEnergyP95);

function makeParetoFigure(x, y, names, colors, markers, yLabelText, ...
    fileStem, showMissingEnergyNote, outDir)

    valid = isfinite(x) & isfinite(y);
    xv = x(valid);
    yv = y(valid);
    validIndices = find(valid);

    % Determine the nondominated set for simultaneous minimization of x/y.
    [xSorted, order] = sort(xv, 'ascend');
    ySorted = yv(order);
    frontMask = false(size(xSorted));
    bestY = inf;
    for k = 1:numel(xSorted)
        if ySorted(k) < bestY
            frontMask(k) = true;
            bestY = ySorted(k);
        end
    end
    xFront = xSorted(frontMask);
    yFront = ySorted(frontMask);

    fig = figure('Color', 'w', 'Units', 'centimeters', ...
        'Position', [2, 2, 15.5, 10.5], 'Renderer', 'painters');
    ax = axes(fig);
    hold(ax, 'on');

    semilogx(ax, xFront, yFront, '-', ...
        'Color', [0.22, 0.22, 0.22], 'LineWidth', 1.25, ...
        'DisplayName', 'Pareto frontier');

    for k = 1:numel(validIndices)
        i = validIndices(k);
        semilogx(ax, x(i), y(i), markers{i}, ...
            'LineStyle', 'none', 'MarkerSize', 8.5, ...
            'MarkerFaceColor', colors(i, :), ...
            'MarkerEdgeColor', [0.08, 0.08, 0.08], ...
            'LineWidth', 0.9, 'HandleVisibility', 'off');
    end

    % Direct labels avoid an oversized legend and remain editable in SVG.
    if contains(fileStem, 'sequence')
        labelXFactor = [1.18, 1.20, 0.78, 1.18, 0.78];
        labelYOffset = [0.35, 0.35, 0.35, -0.75, -0.50];
        align = {'left', 'left', 'right', 'left', 'right'};
    else
        labelXFactor = [1.18, 1.20, 1.00, 1.18, 0.78];
        labelYOffset = [0.15, 0.10, 0.00, -0.22, -0.08];
        align = {'left', 'left', 'center', 'left', 'right'};
    end

    for k = 1:numel(validIndices)
        i = validIndices(k);
        text(ax, x(i) * labelXFactor(i), y(i) + labelYOffset(i), names{i}, ...
            'FontName', 'Times New Roman', 'FontSize', 9.5, ...
            'HorizontalAlignment', align{i}, 'VerticalAlignment', 'middle', ...
            'Interpreter', 'none', 'Color', [0.08, 0.08, 0.08]);
    end

    if showMissingEnergyNote
        text(ax, 0.98, 0.56, 'MagLearn 2 [15]: energy P95 not reported', ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'FontName', 'Times New Roman', ...
            'FontSize', 9, 'FontAngle', 'italic', ...
            'Color', [0.30, 0.30, 0.30]);
    end

    ax.XScale = 'log';
    ax.XLim = [50, 4.0e6];
    yPadding = 0.10 * (max(yv) - min(yv));
    if yPadding == 0
        yPadding = 1;
    end
    ax.YLim = [max(0, min(yv) - yPadding), max(yv) + 1.4 * yPadding];
    ax.XTick = [1e2, 1e3, 1e4, 1e5, 1e6];
    ax.XTickLabel = {'10^2', '10^3', '10^4', '10^5', '10^6'};
    ax.FontName = 'Times New Roman';
    ax.FontSize = 10;
    ax.LineWidth = 0.9;
    ax.Box = 'on';
    ax.TickDir = 'in';
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'off';
    ax.GridColor = [0.72, 0.72, 0.72];
    ax.MinorGridColor = [0.85, 0.85, 0.85];
    ax.GridAlpha = 0.55;
    ax.MinorGridAlpha = 0.38;
    grid(ax, 'on');

    xlabel(ax, 'Average number of trainable parameters (log scale)', ...
        'FontName', 'Times New Roman', 'FontSize', 11);
    ylabel(ax, yLabelText, 'FontName', 'Times New Roman', 'FontSize', 11);
    legend(ax, 'Location', 'northeast', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 9);

    set(fig, 'PaperPositionMode', 'auto');
    drawnow;

    savefig(fig, fullfile(outDir, [fileStem, '.fig']));
    % MATLAB R2024b exportgraphics does not accept SVG on all installations;
    % print -dsvg with the Painters renderer preserves vector editability.
    print(fig, fullfile(outDir, [fileStem, '.svg']), '-dsvg', '-painters');
    exportgraphics(fig, fullfile(outDir, [fileStem, '.png']), ...
        'Resolution', 300, 'BackgroundColor', 'white');

    % EMF is convenient for ungrouping in desktop Visio on Windows.
    try
        exportgraphics(fig, fullfile(outDir, [fileStem, '.emf']), ...
            'ContentType', 'vector', 'BackgroundColor', 'white');
    catch
        try
            print(fig, fullfile(outDir, [fileStem, '.emf']), '-dmeta', '-painters');
        catch
            warning('EMF export was unavailable; the SVG remains fully vector-based.');
        end
    end

    close(fig);
end

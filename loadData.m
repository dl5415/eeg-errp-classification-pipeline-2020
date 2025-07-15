function [signals, event, header] = loadData(path)
dirInfo = dir([path '/*.gdf']);

dirNames = {dirInfo.name};
dirFolders = {dirInfo.folder};
signals = [];
event.type = []; event.position = []; event.name = {};
actionClass = [];
for i = 1:length(dirNames)
    exName = dirNames{i};
    exFolder = dirFolders{i};
    fileName = [exFolder '/' exName];
    [tempSignal, header] = sload(fileName);
    
    load([fileName(1:end-4) '.mat']); 
    gridsize = runData.PM.gridsize;
    endGoals = find(runData.PM.currentState(1,:) == -1);
    runData.PM.currentState(:,endGoals-1) = [];
    goalsReached = find(runData.PM.currentAction == -1);
    
    endGoals = find(runData.PM.currentGoal(1,:) == -1);
    runData.PM.currentGoal(:,endGoals) = [];
    
    if (size(runData.PM.currentState,2) == length(runData.PM.currentAction)+1)
        runData.PM.currentState(:,end) = [];
    end
    
    idxGoal = 1;
    goalsEachStep = [];
    for i=1:length(runData.PM.currentAction)
        if ((idxGoal <= length(goalsReached)) && (i == goalsReached(idxGoal)) && (i<length(runData.PM.currentAction)))
            idxGoal = idxGoal+1;
        end
        goalsEachStep(:,end+1) = runData.PM.currentGoal(:,idxGoal);
    end
    
    idx_letter = 1;
    idx_actions = find(header.EVENT.TYP >= runData.constants.TID_ACTION.value & header.EVENT.TYP <= runData.constants.TID_ACTION.value + runData.constants.NUM_ACTIONS.value);
    for a=1:length(runData.PM.currentAction)
        if (runData.PM.currentAction(a) == -1)
            idx_letter = idx_letter+1;
        else
            currentState = runData.PM.currentState(:,a);
            currentAction = runData.PM.currentAction(:,a);
            if (runData.PM.diagonal_mov_allowed)
                labels = build_optimal_policy_diag(gridsize, goalsEachStep(:,a), runData.constants);
            else
                labels = build_optimal_policy(gridsize, goalsEachStep(:,a), runData.constants);
            end
            actionClass(end+1) = labels(currentState(1), currentState(2), currentAction);
                        
            if (actionClass(end) == -1)
                event.name{end+1} = 'error_sw';
                event.type(end+1) = 1;

            elseif (actionClass(end) == +1)
                event.name{end+1} = 'correct_sw';
                event.type(end+1) = 0;
            else
                error('Unknown event');
            end
            event.position(end+1) = size(signals,1) + header.EVENT.POS(idx_actions(1));
            idx_actions(1) = [];
        end
    end
    signals = cat(1, signals, tempSignal);
end
end
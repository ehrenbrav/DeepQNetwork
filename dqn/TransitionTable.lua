--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'

local trans = torch.class('dqn.TransitionTable')


function trans:__init(args)
    self.stateDim = args.stateDim
    self.numActions = args.numActions
    self.histLen = args.histLen
    self.maxSize = args.maxSize or 1024^2
    self.bufferSize = args.bufferSize or 1024
    self.histType = args.histType or "linear"
    self.histSpacing = args.histSpacing or 1
    self.zeroFrames = args.zeroFrames or 1
    self.nonTermProb = args.nonTermProb or 1
    self.nonEventProb = args.nonEventProb or 1
    self.gpu = args.gpu
    self.numEntries = 0
    self.insertIndex = 0

    self.histIndices = {}
    local histLen = self.histLen
    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
        self.recentMemSize = 2^(histLen-1)
        self.histIndices[1] = 1
        for i=1,histLen-1 do
            self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
        end
    elseif self.histType == "exp1.25" then
        -- The ith history frame is from 1.25^(i-1) frames ago.
        self.histIndices[histLen] = 1
        for i=histLen-1,1,-1 do
            self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
        end
        self.recentMemSize = self.histIndices[1]
        for i=1,histLen do
            self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
        end
    end

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    local s_size = self.stateDim*histLen
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end


function trans:reset()
    self.numEntries = 0
    self.insertIndex = 0
end


function trans:size()
    return self.numEntries
end


function trans:empty()
    return self.numEntries == 0
end


function trans:fill_buffer()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        local s, a, r, s2, term = self:sample_one(1)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
    end
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s2 = self.buf_s2:float():div(255)
    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s)
        self.gpu_s2:copy(self.buf_s2)
    end
end

-- Returns one randomly sampled experience from memory. 
-- Consider setting a non-zero probability that we discard termal
-- or non-reward experiences. 
function trans:sample_one()
    assert(self.numEntries > 1)
    local index
    local valid = false
    
    -- Loop to find a valid experience to send back.
    while not valid do
    
        -- start at 2 because of previous action
        -- Grab a random experience from the memory table.
        index = torch.random(2, self.numEntries-self.recentMemSize)
        
        
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true
        end
        
        -- If nonTermProb is set to less than 1, there is a chance
        -- we discard terminal experiences. Not sure why we'd want to do this...
        if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Note that this is the terminal flag for s_{t+1}.
            valid = false
        end
        
        -- If nonEventProb is set to less than one, there is a chance
        -- we discard experiences not resulting in rewards. Would this accelerate learning?
        if self.nonEventProb < 1 and self.r[index+self.recentMemSize-1] == 0 and
            torch.uniform() > self.nonEventProb then
            -- probability (1-nonEventProb).
            valid = false
        end
    end

    return self:get(index)
end


function trans:sample(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        self:fill_buffer()
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
    end

    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range]
end


function trans:concatFrames(index, use_recent)

    -- Should we use the recent state tables?
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self.t
    end

    local fullstate = s[1].new()
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

    local zero_out = false
    local episode_start = self.histLen

    -- Zero out any frames occuring after a terminal frame.
    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    -- If there are no zero frames, copy the hist_len most recent frames.
    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        fullstate[i]:copy(s[index+self.histIndices[i]-1])
    end

    return fullstate
end


function trans:concatActions(index, use_recent)
    local act_hist = torch.FloatTensor(self.histLen, self.numActions)
    if use_recent then
        a, t = self.recent_a, self.recent_t
    else
        a, t = self.a, self.t
    end

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            act_hist[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        act_hist[i]:copy(self.action_encodings[a[index+self.histIndices[i]-1]])
    end

    return act_hist
end

-- Get the hist_len most recent frames.
function trans:get_recent()
    -- Assumes that the most recent state has been added, but the action has not
    return self:concatFrames(1, true):float():div(255)
end

-- Return a full state in a given index: (s, a, r, s2, terminal).
function trans:get(index)
    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index+1)
    local ar_index = index+self.recentMemSize-1

    return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1]
end

-- Add a new experience.
function trans:add(s, a, r, term)
    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- Incremement the memory counter.
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
        
        -- Spam the console.
        if self.numEntries % 1000 == 0 then
          print("Recorded experiences: " .. self.numEntries)
        end
        if self.numEntries == self.maxSize then
          print("Filled up the experience record...")
        end
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1
    
    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone():float():mul(255)
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    
    -- Record whether this was a terminal experience.
    if term then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end
end


function trans:add_recent_state(s, term)

    -- Process...
    local s = s:clone():float():mul(255):byte()
    
    -- Handle no recent states.
    if #self.recent_s == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_s, s:clone():zero())
            table.insert(self.recent_t, 1)
        end
    end
    
    -- Record the state in the recent_s list.
    table.insert(self.recent_s, s)
    
    -- Record whether this was a terminal state in the recent_t list.
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Kick out old recent states.
    -- recentMemSize is equal to the hist_len * hist_spacing
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end


function trans:add_recent_action(a)
    if #self.recent_a == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_a, 1)
        end
    end

    table.insert(self.recent_a, a)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
    end
end


--[[
Override the write function to serialize this class into a file.
We do not want to store anything into the file, just the necessary info
to create an empty transition table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:write(file)
    file:writeObject({self.stateDim,
                      self.numActions,
                      self.histLen,
                      self.maxSize,
                      self.bufferSize,
                      self.numEntries,
                      self.insertIndex,
                      self.recentMemSize,
                      self.histIndices})
end


--[[
Override the read function to desearialize this class from file.
Recreates an empty table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:read(file)
    local stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = unpack(file:readObject())
    self.stateDim = stateDim
    self.numActions = numActions
    self.histLen = histLen
    self.maxSize = maxSize
    self.bufferSize = bufferSize
    self.recentMemSize = recentMemSize
    self.histIndices = histIndices
    self.numEntries = 0
    self.insertIndex = 0

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
    end
end

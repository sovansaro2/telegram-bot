-- Drop the now-unused `status` (user plan: free/premium) column from `users`.
-- The premium/payment feature was removed from the bot.
-- Uses IF EXISTS so it is safe whether or not the column exists.
ALTER TABLE users DROP COLUMN IF EXISTS status;
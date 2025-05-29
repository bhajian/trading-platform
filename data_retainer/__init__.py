"""
data_retainer
=============

• Watches Redis lists created by *data_loader*.
• Whenever a list grows past its configured window, pops the oldest
  rows, converts them from JSON → CSV, and appends them to disk.
• Maintains separate history files for raw and augmented data:
      history/raw/<SYM>_raw.csv
      history/augmented/<SYM>_aug.csv
"""

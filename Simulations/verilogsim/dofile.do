add wave -position insertpoint  \
sim:/tb_signal_analyser/A \
sim:/tb_signal_analyser/initdone \
sim:/tb_signal_analyser/clock \
sim:/tb_signal_analyser/Z \

run -all

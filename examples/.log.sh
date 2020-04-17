#!/bin/bash
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$SOURCE_DIR/.log"
GPU="$DIR/gpu.csv"
RUNNING="$DIR/.running"

while [ -n "$1" ]; do
  case "$1" in
    -n)
      SLEEP="$2"
      shift
      ;;
    stop)
      rm "$RUNNING"
      exit 0
      ;;
    clear)
      rm "$DIR"/*
      exit 0
      ;;
    *) echo "Option $1 not recognized" ;;
  esac
  shift
done

if [ ! -d "$DIR" ]; then
  mkdir "$DIR"
fi

if [ -f "$RUNNING" ]; then
  exit 0
else
  touch "$RUNNING"
fi

if [ ! -f "$GPU" ]; then
  eval "nvidia-smi --query-gpu=timestamp,index,fan.speed,pstate,clocks_throttle_reasons.active,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem --format=csv > $GPU"
fi

if [ -z "$SLEEP" ]; then
  SLEEP="1"
fi

while [ -f "$RUNNING" ]; do
  sleep "$SLEEP"
  eval "nvidia-smi --query-gpu=timestamp,index,fan.speed,pstate,clocks_throttle_reasons.active,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.sm,clocks.mem --format=csv,noheader >> $GPU"
done

"use client";

import { motion, type Variants } from "framer-motion";
import { cn } from "@/lib/utils";
import { type ReactNode, type RefObject } from "react";

type TimelineContentProps = {
  animationNum?: number;
  timelineRef?: RefObject<HTMLElement | null>;
  customVariants?: Variants;
  className?: string;
  children?: ReactNode;
};

export function TimelineContent({
  animationNum = 0,
  timelineRef,
  customVariants,
  className,
  children,
}: TimelineContentProps) {
  return (
    <motion.div
      className={cn(className)}
      variants={customVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{
        once: true,
        amount: 0.25,
        root: timelineRef,
      }}
      custom={animationNum}
    >
      {children}
    </motion.div>
  );
}

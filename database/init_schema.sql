-- MySQL Script generated by MySQL Workbench
-- Sat Apr 23 16:14:55 2022
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema robustar
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema robustar
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `robustar` DEFAULT CHARACTER SET utf8 ;
USE `robustar` ;

-- -----------------------------------------------------
-- Table `robustar`.`split`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`split` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `split_name` VARCHAR(45) NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`visu_rel`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`visu_rel` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `img_id` INT NULL,
  `visu_type` VARCHAR(45) NULL COMMENT 'Can be ',
  `visu_path` VARCHAR(45) NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`paired_set`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`paired_set` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `img_path` VARCHAR(256) NULL,
  `train_id` INT NULL,
  `visu_rel_id` INT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE,
  INDEX `paired_visu_rel_idx` (`visu_rel_id` ASC) VISIBLE,
  CONSTRAINT `paired_visu_rel`
    FOREIGN KEY (`visu_rel_id`)
    REFERENCES `robustar`.`visu_rel` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`train_set`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`train_set` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `img_path` VARCHAR(256) NULL,
  `annotated` TINYINT NULL,
  `paired_id` INT NULL,
  `visu_rel_id` INT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE,
  INDEX `train-paired_idx` (`paired_id` ASC) VISIBLE,
  INDEX `train-visu_idx` (`visu_rel_id` ASC) VISIBLE,
  CONSTRAINT `train-paired`
    FOREIGN KEY (`paired_id`)
    REFERENCES `robustar`.`paired_set` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `train-visu`
    FOREIGN KEY (`visu_rel_id`)
    REFERENCES `robustar`.`visu_rel` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`influ_rel`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`influ_rel` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `influ_path` VARCHAR(256) NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`val_set`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`val_set` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `classified` TINYINT NULL,
  `influ_rel_id` INT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE,
  INDEX `val_influ_rel_idx` (`influ_rel_id` ASC) VISIBLE,
  CONSTRAINT `val_influ_rel`
    FOREIGN KEY (`influ_rel_id`)
    REFERENCES `robustar`.`influ_rel` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`test_set`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`test_set` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `classified` TINYINT NULL,
  `influ_rel_id` INT NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE,
  INDEX `test_influ_rel_idx` (`influ_rel_id` ASC) VISIBLE,
  CONSTRAINT `test_influ_rel`
    FOREIGN KEY (`influ_rel_id`)
    REFERENCES `robustar`.`influ_rel` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `robustar`.`model`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `robustar`.`model` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(256) NULL,
  `type` VARCHAR(256) NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;